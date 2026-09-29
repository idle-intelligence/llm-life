//! Raw wgpu + hand-written WGSL, same shape as llm-web's `lean` crate: no
//! framework graph, async-only readback (`read_buffer_u32`'s doc comment),
//! ping-pong buffers, one command encoder per timed batch of generations.
//! This benchmark only runs natively, but the engine avoids any native-only
//! API in the shared path so the same kernels stay portable to a wasm32 +
//! WebGPU build later.

use std::borrow::Cow;
use std::time::{Duration, Instant};

use bytemuck::{Pod, Zeroable};

pub struct Engine {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub lut_pipeline: wgpu::ComputePipeline,
    pub bitpack_pipeline: wgpu::ComputePipeline,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct LutParams {
    width: u32,
    height: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct BitpackParams {
    words_per_row: u32,
    height: u32,
    birth_mask: u32,
    survive_mask: u32,
}

fn make_pipeline(device: &wgpu::Device, label: &str, src: &str) -> wgpu::ComputePipeline {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(src)),
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: None,
        module: &module,
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    })
}

impl Engine {
    pub async fn new_async() -> anyhow::Result<Self> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .await
            .map_err(|e| anyhow::anyhow!("no wgpu adapter: {e}"))?;
        let adapter_limits = adapter.limits();
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("life-lut-bench"),
                required_features: wgpu::Features::empty(),
                required_limits: adapter_limits,
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
            })
            .await
            .map_err(|e| anyhow::anyhow!("no wgpu device: {e}"))?;

        Ok(Engine {
            lut_pipeline: make_pipeline(&device, "lut_byte", include_str!("shaders/lut_byte.wgsl")),
            bitpack_pipeline: make_pipeline(&device, "bitpack", include_str!("shaders/bitpack.wgsl")),
            device,
            queue,
        })
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn new() -> anyhow::Result<Self> {
        pollster::block_on(Self::new_async())
    }

    pub fn adapter_info_string(adapter: &wgpu::Adapter) -> String {
        let info = adapter.get_info();
        format!("{} ({:?}, {:?})", info.name, info.backend, info.device_type)
    }

    pub fn max_storage_buffer_binding_size(&self) -> u64 {
        self.device.limits().max_storage_buffer_binding_size as u64
    }

    pub fn buf_u32(&self, data: &[u32], label: &str) -> wgpu::Buffer {
        use wgpu::util::DeviceExt;
        self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::cast_slice(data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        })
    }

    pub fn buf_empty_u32(&self, len: usize, label: &str) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: (len.max(1) * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
    }

    fn buf_uniform<T: Pod>(&self, data: T, label: &str) -> wgpu::Buffer {
        use wgpu::util::DeviceExt;
        self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::bytes_of(&data),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        })
    }

    /// One copy-to-staging + map + read. Never the sync `.into_data()`
    /// equivalent in a WASM build (deadlocks the browser) — this async path
    /// is the only readback this crate uses, matching llm-web's `lean`
    /// engine's `read_buffer` rule.
    pub async fn read_buffer_u32(&self, buf: &wgpu::Buffer, len: usize) -> Vec<u32> {
        let size = (len * 4) as u64;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback_staging"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("readback") });
        encoder.copy_buffer_to_buffer(buf, 0, &staging, 0, size);
        self.queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = futures_channel::oneshot::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        #[cfg(not(target_arch = "wasm32"))]
        self.device.poll(wgpu::PollType::Wait).expect("device poll failed");
        rx.await.expect("map_async channel dropped").expect("buffer map failed");
        let data = slice.get_mapped_range();
        let out = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging.unmap();
        out
    }

    fn dispatch_1d(&self, pipeline: &wgpu::ComputePipeline, bind_group: &wgpu::BindGroup, n: u32) -> wgpu::CommandEncoder {
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("gen") });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("gen"), timestamp_writes: None });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            let workgroups = n.div_ceil(256);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        encoder
    }

    /// Runs `generations` LUT steps with ping-pong buffers, timing only the
    /// steady-state loop (after `warmup` untimed generations). No readback
    /// inside the timed loop; the caller reads back once at the end. Command
    /// buffers are flushed every `flush_every` generations with a blocking
    /// native poll (see llm-web's `lean::Engine::flush_encoder` — an
    /// unbounded backlog of un-retired submissions hits wgpu-core's 60s
    /// cleanup-wait timeout on native).
    #[allow(clippy::too_many_arguments)]
    pub fn run_lut_timed(&self, width: u32, height: u32, lut: &[u32; 512], initial: &[u32], warmup: u32, generations: u32, flush_every: u32) -> (Vec<u32>, Duration) {
        let n = width * height;
        let params = self.buf_uniform(LutParams { width, height }, "lut_params");
        let lut_buf = self.buf_u32(lut, "lut_table");
        let mut a = self.buf_u32(initial, "grid_a");
        let mut b = self.buf_empty_u32(n as usize, "grid_b");

        let bg = |src: &wgpu::Buffer, dst: &wgpu::Buffer| {
            let layout = self.lut_pipeline.get_bind_group_layout(0);
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: params.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: lut_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: src.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: dst.as_entire_binding() },
                ],
            })
        };

        for gen in 0..warmup {
            let bind_group = bg(&a, &b);
            let encoder = self.dispatch_1d(&self.lut_pipeline, &bind_group, n);
            self.queue.submit(Some(encoder.finish()));
            std::mem::swap(&mut a, &mut b);
            if gen % flush_every == flush_every - 1 {
                #[cfg(not(target_arch = "wasm32"))]
                let _ = self.device.poll(wgpu::PollType::Wait);
            }
        }
        #[cfg(not(target_arch = "wasm32"))]
        let _ = self.device.poll(wgpu::PollType::Wait);

        let start = Instant::now();
        for gen in 0..generations {
            let bind_group = bg(&a, &b);
            let encoder = self.dispatch_1d(&self.lut_pipeline, &bind_group, n);
            self.queue.submit(Some(encoder.finish()));
            std::mem::swap(&mut a, &mut b);
            if gen % flush_every == flush_every - 1 {
                #[cfg(not(target_arch = "wasm32"))]
                let _ = self.device.poll(wgpu::PollType::Wait);
            }
        }
        #[cfg(not(target_arch = "wasm32"))]
        let _ = self.device.poll(wgpu::PollType::Wait);
        let elapsed = start.elapsed();

        let out = pollster::block_on(self.read_buffer_u32(&a, n as usize));
        (out, elapsed)
    }

    /// Same shape as `run_lut_timed` but for the bit-packed adder-network
    /// kernel: `words_per_row = width / 32` (see `bitpack.wgsl`'s doc
    /// comment — caller must ensure `width % 32 == 0`).
    #[allow(clippy::too_many_arguments)]
    pub fn run_bitpack_timed(&self, words_per_row: u32, height: u32, birth_mask: u32, survive_mask: u32, initial: &[u32], warmup: u32, generations: u32, flush_every: u32) -> (Vec<u32>, Duration) {
        let n = words_per_row * height;
        let params = self.buf_uniform(BitpackParams { words_per_row, height, birth_mask, survive_mask }, "bitpack_params");
        let mut a = self.buf_u32(initial, "words_a");
        let mut b = self.buf_empty_u32(n as usize, "words_b");

        let bg = |src: &wgpu::Buffer, dst: &wgpu::Buffer| {
            let layout = self.bitpack_pipeline.get_bind_group_layout(0);
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: params.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: src.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: dst.as_entire_binding() },
                ],
            })
        };

        for gen in 0..warmup {
            let bind_group = bg(&a, &b);
            let encoder = self.dispatch_1d(&self.bitpack_pipeline, &bind_group, n);
            self.queue.submit(Some(encoder.finish()));
            std::mem::swap(&mut a, &mut b);
            if gen % flush_every == flush_every - 1 {
                #[cfg(not(target_arch = "wasm32"))]
                let _ = self.device.poll(wgpu::PollType::Wait);
            }
        }
        #[cfg(not(target_arch = "wasm32"))]
        let _ = self.device.poll(wgpu::PollType::Wait);

        let start = Instant::now();
        for gen in 0..generations {
            let bind_group = bg(&a, &b);
            let encoder = self.dispatch_1d(&self.bitpack_pipeline, &bind_group, n);
            self.queue.submit(Some(encoder.finish()));
            std::mem::swap(&mut a, &mut b);
            if gen % flush_every == flush_every - 1 {
                #[cfg(not(target_arch = "wasm32"))]
                let _ = self.device.poll(wgpu::PollType::Wait);
            }
        }
        #[cfg(not(target_arch = "wasm32"))]
        let _ = self.device.poll(wgpu::PollType::Wait);
        let elapsed = start.elapsed();

        let out = pollster::block_on(self.read_buffer_u32(&a, n as usize));
        (out, elapsed)
    }
}
