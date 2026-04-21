// Copyright 2023 Citra Emulator Project
// Copyright 2024 Borked3DS Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include "common/alignment.h"
#include <cstdlib>
#include "common/literals.h"
#include "common/logging/log.h"
#include "common/math_util.h"
#include "common/profiling.h"
#include "common/settings.h"
#include "core/memory.h"
#include "video_core/pica/pica_core.h"
#include "video_core/renderer_vulkan/renderer_vulkan.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_rasterizer.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/texture/texture_decode.h"

namespace Vulkan {

std::atomic<u64> g_vk_draw_counter{0};

namespace {

using TriangleTopology = Pica::PipelineRegs::TriangleTopology;
using VideoCore::SurfaceType;

using namespace Common::Literals;
using namespace Pica::Shader::Generator;

constexpr u64 STREAM_BUFFER_SIZE = 64_MiB;
constexpr u64 UNIFORM_BUFFER_SIZE = 4_MiB;
constexpr u64 TEXTURE_BUFFER_SIZE = 2_MiB;

constexpr vk::BufferUsageFlags BUFFER_USAGE =
    vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eIndexBuffer;

struct DrawParams {
    u32 vertex_count;
    s32 vertex_offset;
    u32 binding_count;
    std::array<u32, 16> bindings;
    bool is_indexed;
};

[[nodiscard]] u64 TextureBufferSize(const Instance& instance) {
    // Use the smallest texel size from the texel views
    // which corresponds to eR32G32Sfloat
    const u64 max_size = instance.MaxTexelBufferElements() * 8;
    return std::min(max_size, TEXTURE_BUFFER_SIZE);
}

[[nodiscard]] bool IsValidImageView(const vk::ImageView view) {
    return static_cast<bool>(view);
}

[[nodiscard]] bool IsDrawTraceEnabled() {
    const char* value = std::getenv("BORKED3DS_V3DV_TRACE_DRAW");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

[[nodiscard]] bool IsStrictCompatEnabled() {
    const char* value = std::getenv("BORKED3DS_V3DV_STRICT_COMPAT");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

std::atomic<u64> g_vk_software_bypass_counter{0};
std::atomic<u64> g_vk_textured_software_bypass_counter{0};
std::atomic<u64> g_vk_large_textured_software_allow_counter{0};
std::atomic<u64> g_vk_batch42_textured_software_skip_counter{0};
std::atomic<u64> g_vk_nonindexed96_textured_software_skip_counter{0};
std::atomic<u64> g_vk_nonindexed36_textured_software_skip_counter{0};
std::atomic<u64> g_vk_non_bypassed_software_trace_counter{0};
std::atomic<u64> g_vk_medium_textured_software_skip_counter{0};
std::atomic<u64> g_vk_startup_textured_software_skip_counter{0};

[[nodiscard]] bool ArePrimaryTexturesDisabled(const Pica::RegsInternal& regs) {
    const auto& textures = regs.texturing.GetTextures();
    for (u32 i = 0; i < 3; ++i) {
        if (textures[i].enabled) {
            return false;
        }
    }
    return true;
}

[[nodiscard]] bool HasPrimaryTexturesEnabled(const Pica::RegsInternal& regs) {
    return !ArePrimaryTexturesDisabled(regs);
}

[[nodiscard]] u32 CountEnabledPrimaryTextures(const Pica::RegsInternal& regs) {
    const auto& textures = regs.texturing.GetTextures();
    u32 enabled = 0;
    for (u32 i = 0; i < 3; ++i) {
        enabled += textures[i].enabled ? 1u : 0u;
    }
    return enabled;
}

[[nodiscard]] bool HasSinglePrimaryTexture0Format8(const Pica::RegsInternal& regs) {
    const auto& textures = regs.texturing.GetTextures();
    return textures[0].enabled && !textures[1].enabled && !textures[2].enabled &&
           static_cast<u32>(textures[0].format) == 8u;
}

[[nodiscard]] bool HasActiveDepthState(const Pica::RegsInternal& regs) {
    return regs.framebuffer.output_merger.depth_test_enable != 0 ||
           regs.framebuffer.output_merger.depth_write_enable != 0;
}

[[nodiscard]] bool ShouldBypassFragileSoftwareDraw(const Pica::RegsInternal& regs,
                                                   std::size_t vertex_batch_size) {
    if (!IsStrictCompatEnabled()) {
        return false;
    }
    if (vertex_batch_size == 0 || vertex_batch_size > 24) {
        return false;
    }
    if (regs.pipeline.num_vertices == 0 || regs.pipeline.num_vertices > 24) {
        return false;
    }
    if (!ArePrimaryTexturesDisabled(regs)) {
        return false;
    }
    if (regs.framebuffer.IsShadowRendering()) {
        return false;
    }
    return true;
}

[[nodiscard]] bool ShouldBypassFragileTexturedSoftwareDraw(const Pica::RegsInternal& regs,
                                                           std::size_t vertex_batch_size) {
    if (!IsStrictCompatEnabled()) {
        return false;
    }
    // Pi 5 / V3DV startup textured-software workaround:
    // catch the first modest textured draws that were exposed only after the PICA-side
    // startup fallback, while staying conservative enough to avoid broad regressions.
    if (vertex_batch_size == 0 || vertex_batch_size > 96) {
        return false;
    }
    if (regs.pipeline.num_vertices == 0 || regs.pipeline.num_vertices > 96) {
        return false;
    }
    if (!HasPrimaryTexturesEnabled(regs)) {
        return false;
    }
    if (CountEnabledPrimaryTextures(regs) > 1) {
        return false;
    }
    if (regs.framebuffer.IsShadowRendering()) {
        return false;
    }
    if (HasActiveDepthState(regs)) {
        return false;
    }
    return true;
}

[[nodiscard]] bool ShouldAttemptTinyTexturedSoftwareDraw(const Pica::RegsInternal& regs,
                                                         std::size_t vertex_batch_size) {
    if (!IsStrictCompatEnabled()) {
        return false;
    }
    if (vertex_batch_size != 6) {
        return false;
    }
    if (regs.pipeline.num_vertices != 6) {
        return false;
    }
    if (!HasPrimaryTexturesEnabled(regs)) {
        return false;
    }
    if (CountEnabledPrimaryTextures(regs) != 1) {
        return false;
    }
    if (regs.framebuffer.IsShadowRendering()) {
        return false;
    }
    if (HasActiveDepthState(regs)) {
        return false;
    }
    return true;
}


[[nodiscard]] bool ShouldAttemptMediumTexturedSoftwareDraw(const Pica::RegsInternal& regs,
                                                           std::size_t vertex_batch_size) {
    if (!IsStrictCompatEnabled()) {
        return false;
    }
    if (vertex_batch_size < 12 || vertex_batch_size > 96 || vertex_batch_size == 42) {
        return false;
    }
    if (regs.pipeline.num_vertices < 12 || regs.pipeline.num_vertices > 96 ||
        regs.pipeline.num_vertices == 42) {
        return false;
    }
    if (!HasPrimaryTexturesEnabled(regs)) {
        return false;
    }
    if (CountEnabledPrimaryTextures(regs) != 1) {
        return false;
    }
    if (regs.framebuffer.IsShadowRendering()) {
        return false;
    }
    if (HasActiveDepthState(regs)) {
        return false;
    }
    return true;
}

[[nodiscard]] bool ShouldSkipStartupTexturedSoftwareDraw(const Pica::RegsInternal& regs,
                                                      std::size_t vertex_batch_size) {
    if (!IsStrictCompatEnabled()) {
        return false;
    }
    if (vertex_batch_size == 0 || vertex_batch_size > 320) {
        return false;
    }
    if (regs.pipeline.num_vertices == 0 || regs.pipeline.num_vertices > 320) {
        return false;
    }
    if (!HasPrimaryTexturesEnabled(regs)) {
        return false;
    }
    if (CountEnabledPrimaryTextures(regs) != 1) {
        return false;
    }
    if (regs.framebuffer.IsShadowRendering()) {
        return false;
    }
    if (HasActiveDepthState(regs)) {
        return false;
    }
    return true;
}

[[nodiscard]] bool ShouldAttemptLargeTexturedSoftwareDraw(const Pica::RegsInternal& regs,
                                                          std::size_t vertex_batch_size) {
    if (!IsStrictCompatEnabled()) {
        return false;
    }
    if (vertex_batch_size != 42) {
        return false;
    }
    if (regs.pipeline.num_vertices != 42) {
        return false;
    }
    if (!HasPrimaryTexturesEnabled(regs)) {
        return false;
    }
    if (CountEnabledPrimaryTextures(regs) != 1) {
        return false;
    }
    if (regs.framebuffer.IsShadowRendering()) {
        return false;
    }
    if (HasActiveDepthState(regs)) {
        return false;
    }
    return true;
}

[[nodiscard]] bool ShouldSkipBatch42TexturedSoftwareDraw(const Pica::RegsInternal& regs,
                                                         std::size_t vertex_batch_size) {
    return ShouldAttemptLargeTexturedSoftwareDraw(regs, vertex_batch_size);
}

[[nodiscard]] bool ShouldSkipNonIndexed96TexturedSoftwareDraw(const Pica::RegsInternal& regs,
                                                              std::size_t vertex_batch_size) {
    if (!IsStrictCompatEnabled()) {
        return false;
    }
    if (vertex_batch_size != 96) {
        return false;
    }
    if (regs.pipeline.num_vertices != 96) {
        return false;
    }
    if (!HasPrimaryTexturesEnabled(regs)) {
        return false;
    }
    if (CountEnabledPrimaryTextures(regs) != 1) {
        return false;
    }
    if (regs.framebuffer.IsShadowRendering()) {
        return false;
    }
    if (HasActiveDepthState(regs)) {
        return false;
    }
    return true;
}

[[nodiscard]] bool ShouldSkipNonIndexed36TexturedSoftwareDraw(const Pica::RegsInternal& regs,
                                                              std::size_t vertex_batch_size) {
    if (!IsStrictCompatEnabled()) {
        return false;
    }
    if (vertex_batch_size != 36) {
        return false;
    }
    if (regs.pipeline.num_vertices != 36) {
        return false;
    }
    if (!HasSinglePrimaryTexture0Format8(regs)) {
        return false;
    }
    if (regs.framebuffer.IsShadowRendering()) {
        return false;
    }
    if (HasActiveDepthState(regs)) {
        return false;
    }
    return true;
}

} // Anonymous namespace

RasterizerVulkan::RasterizerVulkan(Memory::MemorySystem& memory, Pica::PicaCore& pica,
                                   VideoCore::CustomTexManager& custom_tex_manager,
                                   VideoCore::RendererBase& renderer,
                                   Frontend::EmuWindow& emu_window, const Instance& instance,
                                   Scheduler& scheduler, RenderManager& renderpass_cache,
                                   DescriptorUpdateQueue& update_queue_, u32 image_count)
    : RasterizerAccelerated{memory, pica}, instance{instance}, scheduler{scheduler},
      renderpass_cache{renderpass_cache}, update_queue{update_queue_},
      pipeline_cache{instance, scheduler, renderpass_cache, update_queue},
      runtime{instance, scheduler, renderpass_cache, update_queue, image_count},
      res_cache{memory, custom_tex_manager, runtime, regs, renderer},
      stream_buffer{instance, scheduler, BUFFER_USAGE, STREAM_BUFFER_SIZE},
      uniform_buffer{instance, scheduler, vk::BufferUsageFlagBits::eUniformBuffer,
                     UNIFORM_BUFFER_SIZE},
      texture_buffer{instance, scheduler, vk::BufferUsageFlagBits::eUniformTexelBuffer,
                     TextureBufferSize(instance)},
      texture_lf_buffer{instance, scheduler, vk::BufferUsageFlagBits::eUniformTexelBuffer,
                        TextureBufferSize(instance)},
      async_shaders{Settings::values.async_shader_compilation.GetValue()} {

    vertex_buffers.fill(stream_buffer.Handle());

    // Query uniform buffer alignment.
    uniform_buffer_alignment = instance.UniformMinAlignment();
    uniform_size_aligned_vs_pica =
        Common::AlignUp<u32>(sizeof(VSPicaUniformData), uniform_buffer_alignment);
    uniform_size_aligned_vs = Common::AlignUp<u32>(sizeof(VSUniformData), uniform_buffer_alignment);
    uniform_size_aligned_fs = Common::AlignUp<u32>(sizeof(FSUniformData), uniform_buffer_alignment);

    // Define vertex layout for software shaders
    MakeSoftwareVertexLayout();
    pipeline_info.vertex_layout = software_layout;


    const vk::Device device = instance.GetDevice();
    texture_lf_view = device.createBufferViewUnique({
        .buffer = texture_lf_buffer.Handle(),
        .format = vk::Format::eR32G32Sfloat,
        .offset = 0,
        .range = VK_WHOLE_SIZE,
    });
    texture_rg_view = device.createBufferViewUnique({
        .buffer = texture_buffer.Handle(),
        .format = vk::Format::eR32G32Sfloat,
        .offset = 0,
        .range = VK_WHOLE_SIZE,
    });
    texture_rgba_view = device.createBufferViewUnique({
        .buffer = texture_buffer.Handle(),
        .format = vk::Format::eR32G32B32A32Sfloat,
        .offset = 0,
        .range = VK_WHOLE_SIZE,
    });

    scheduler.RegisterOnSubmit([&renderpass_cache] { renderpass_cache.EndRendering(); });

    // Prepare the static buffer descriptor set.
    const auto buffer_set = pipeline_cache.Acquire(DescriptorHeapType::Buffer);
    update_queue.AddBuffer(buffer_set, 0, uniform_buffer.Handle(), 0, sizeof(VSPicaUniformData));
    update_queue.AddBuffer(buffer_set, 1, uniform_buffer.Handle(), 0, sizeof(VSUniformData));
    update_queue.AddBuffer(buffer_set, 2, uniform_buffer.Handle(), 0, sizeof(FSUniformData));
    update_queue.AddTexelBuffer(buffer_set, 3, *texture_lf_view);
    update_queue.AddTexelBuffer(buffer_set, 4, *texture_rg_view);
    update_queue.AddTexelBuffer(buffer_set, 5, *texture_rgba_view);

    const auto texture_set = pipeline_cache.Acquire(DescriptorHeapType::Texture);
    Surface& null_surface = res_cache.GetSurface(VideoCore::NULL_SURFACE_ID);
    Sampler& null_sampler = res_cache.GetSampler(VideoCore::NULL_SAMPLER_ID);

    // Prepare texture and utility descriptor sets.
    for (u32 i = 0; i < 3; i++) {
        update_queue.AddImageSampler(texture_set, i, 0, null_surface.ImageView(),
                                     null_sampler.Handle());
    }

    const auto utility_set = pipeline_cache.Acquire(DescriptorHeapType::Utility);
    update_queue.AddStorageImage(utility_set, 0, null_surface.StorageView());
    update_queue.AddImageSampler(utility_set, 1, 0, null_surface.ImageView(),
                                 null_sampler.Handle());
    update_queue.Flush();

    SyncEntireState();
}

RasterizerVulkan::~RasterizerVulkan() = default;

void RasterizerVulkan::TickFrame() {
    res_cache.TickFrame();
}

void RasterizerVulkan::LoadDiskResources(const std::atomic_bool& stop_loading,
                                         const VideoCore::DiskResourceLoadCallback& callback) {
    pipeline_cache.LoadDiskCache();
}

void RasterizerVulkan::SyncFixedState() {
    SyncCullMode();
    SyncBlendEnabled();
    SyncBlendFuncs();
    SyncBlendColor();
    SyncLogicOp();
    SyncStencilTest();
    SyncDepthTest();
    SyncColorWriteMask();
    SyncStencilWriteMask();
    SyncDepthWriteMask();
}

void RasterizerVulkan::SetupVertexArray() {
    const auto [vs_input_index_min, vs_input_index_max, vs_input_size] = vertex_info;
    auto [array_ptr, array_offset, invalidate] = stream_buffer.Map(vs_input_size, 16);

    /**
     * The Nintendo 3DS has 12 attribute loaders which are used to tell the GPU
     * how to interpret vertex data. The program firsts sets GPUREG_ATTR_BUF_BASE to the base
     * address containing the vertex array data. The data for each attribute loader (i) can be found
     * by adding GPUREG_ATTR_BUFi_OFFSET to the base address. Attribute loaders can be thought
     * as something analogous to Vulkan bindings. The user can store attributes in separate loaders
     * or interleave them in the same loader.
     **/
    const auto& vertex_attributes = regs.pipeline.vertex_attributes;
    const PAddr base_address = vertex_attributes.GetPhysicalBaseAddress(); // GPUREG_ATTR_BUF_BASE
    const u32 stride_alignment = instance.GetMinVertexStrideAlignment();

    VertexLayout& layout = pipeline_info.vertex_layout;
    layout.binding_count = 0;
    layout.attribute_count = 16;
    enable_attributes.fill(false);

    u32 buffer_offset = 0;
    for (const auto& loader : vertex_attributes.attribute_loaders) {
        if (loader.component_count == 0 || loader.byte_count == 0) {
            continue;
        }

        // Analyze the attribute loader by checking which attributes it provides
        u32 offset = 0;
        for (u32 comp = 0; comp < loader.component_count && comp < 12; comp++) {
            const u32 attribute_index = loader.GetComponent(comp);
            if (attribute_index >= 12) {
                // Attribute ids 12, to 15 signify 4, 8, 12 and 16-byte paddings respectively.
                offset = Common::AlignUp(offset, 4);
                offset += (attribute_index - 11) * 4;
                continue;
            }

            const u32 size = vertex_attributes.GetNumElements(attribute_index);
            if (size == 0) {
                continue;
            }

            offset =
                Common::AlignUp(offset, vertex_attributes.GetElementSizeInBytes(attribute_index));

            const u32 input_reg = regs.vs.GetRegisterForAttribute(attribute_index);
            const auto format = vertex_attributes.GetFormat(attribute_index);

            VertexAttribute& attribute = layout.attributes[input_reg];
            attribute.binding.Assign(layout.binding_count);
            attribute.location.Assign(input_reg);
            attribute.offset.Assign(offset);
            attribute.type.Assign(format);
            attribute.size.Assign(size);

            enable_attributes[input_reg] = true;
            offset += vertex_attributes.GetStride(attribute_index);
        }

        const PAddr data_addr =
            base_address + loader.data_offset + (vs_input_index_min * loader.byte_count);
        const u32 vertex_num = vs_input_index_max - vs_input_index_min + 1;
        u32 data_size = loader.byte_count * vertex_num;
        res_cache.FlushRegion(data_addr, data_size);

        const MemoryRef src_ref = memory.GetPhysicalRef(data_addr);
        if (src_ref.GetSize() < data_size) {
            LOG_ERROR(Render_Vulkan,
                      "Vertex buffer size {} exceeds available space {} at address {:#016X}",
                      data_size, src_ref.GetSize(), data_addr);
        }

        const u8* src_ptr = src_ref.GetPtr();
        u8* dst_ptr = array_ptr + buffer_offset;

        // Align stride up if required by Vulkan implementation.
        const u32 aligned_stride =
            Common::AlignUp(static_cast<u32>(loader.byte_count), stride_alignment);
        if (aligned_stride == loader.byte_count) {
            std::memcpy(dst_ptr, src_ptr, data_size);
        } else {
            for (std::size_t vertex = 0; vertex < vertex_num; vertex++) {
                std::memcpy(dst_ptr + vertex * aligned_stride, src_ptr + vertex * loader.byte_count,
                            loader.byte_count);
            }
        }

        // Create the binding associated with this loader
        VertexBinding& binding = layout.bindings[layout.binding_count];
        binding.binding.Assign(layout.binding_count);
        binding.fixed.Assign(0);
        binding.stride.Assign(aligned_stride);

        // Keep track of the binding offsets so we can bind the vertex buffer later
        binding_offsets[layout.binding_count++] = static_cast<u32>(array_offset + buffer_offset);
        buffer_offset += Common::AlignUp(aligned_stride * vertex_num, 4);
    }

    stream_buffer.Commit(buffer_offset);

    // Assign the rest of the attributes to the last binding
    SetupFixedAttribs();
}

void RasterizerVulkan::SetupFixedAttribs() {
    const auto& vertex_attributes = regs.pipeline.vertex_attributes;
    VertexLayout& layout = pipeline_info.vertex_layout;

    auto [fixed_ptr, fixed_offset, _] = stream_buffer.Map(16 * sizeof(Common::Vec4f), 0);
    binding_offsets[layout.binding_count] = static_cast<u32>(fixed_offset);

    // Reserve the last binding for fixed and default attributes
    // Place the default attrib at offset zero for easy access
    static const Common::Vec4f default_attrib{0.f, 0.f, 0.f, 1.f};
    std::memcpy(fixed_ptr, default_attrib.AsArray(), sizeof(Common::Vec4f));

    // Find all fixed attributes and assign them to the last binding
    u32 offset = sizeof(Common::Vec4f);
    for (std::size_t i = 0; i < 16; i++) {
        if (vertex_attributes.IsDefaultAttribute(i)) {
            const u32 reg = regs.vs.GetRegisterForAttribute(i);
            if (!enable_attributes[reg]) {
                const auto& attr = pica.input_default_attributes[i];
                const std::array data = {attr.x.ToFloat32(), attr.y.ToFloat32(), attr.z.ToFloat32(),
                                         attr.w.ToFloat32()};

                const u32 data_size = sizeof(float) * static_cast<u32>(data.size());
                std::memcpy(fixed_ptr + offset, data.data(), data_size);

                VertexAttribute& attribute = layout.attributes[reg];
                attribute.binding.Assign(layout.binding_count);
                attribute.location.Assign(reg);
                attribute.offset.Assign(offset);
                attribute.type.Assign(Pica::PipelineRegs::VertexAttributeFormat::FLOAT);
                attribute.size.Assign(4);

                offset += data_size;
                enable_attributes[reg] = true;
            }
        }
    }

    // Loop one more time to find unused attributes and assign them to the default one
    // If the attribute is just disabled, shove the default attribute to avoid
    // errors if the shader ever decides to use it.
    for (u32 i = 0; i < 16; i++) {
        if (!enable_attributes[i]) {
            VertexAttribute& attribute = layout.attributes[i];
            attribute.binding.Assign(layout.binding_count);
            attribute.location.Assign(i);
            attribute.offset.Assign(0);
            attribute.type.Assign(Pica::PipelineRegs::VertexAttributeFormat::FLOAT);
            attribute.size.Assign(4);
        }
    }

    // Define the fixed+default binding
    VertexBinding& binding = layout.bindings[layout.binding_count];
    binding.binding.Assign(layout.binding_count++);
    binding.fixed.Assign(1);
    binding.stride.Assign(offset);

    stream_buffer.Commit(offset);
}

bool RasterizerVulkan::SetupVertexShader() {
    BORKED3DS_PROFILE("Vulkan", "Vertex Shader Setup");
    return pipeline_cache.UseProgrammableVertexShader(regs, pica.vs_setup,
                                                      pipeline_info.vertex_layout, accurate_mul);
}

bool RasterizerVulkan::SetupGeometryShader() {
    BORKED3DS_PROFILE("Vulkan", "Geometry Shader Setup");

    if (regs.pipeline.use_gs != Pica::PipelineRegs::UseGS::No) {
        LOG_ERROR(Render_Vulkan, "Accelerate draw doesn't support geometry shader");
        return false;
    }

    // Enable the quaternion fix-up geometry-shader only if we are actually doing per-fragment
    // lighting and care about proper quaternions. Otherwise just use standard vertex+fragment
    // shaders. We also don't need a geometry shader if the barycentric extension is supported.
    if (regs.lighting.disable || instance.IsFragmentShaderBarycentricSupported()) {
        pipeline_cache.UseTrivialGeometryShader();
        return true;
    }

    return pipeline_cache.UseFixedGeometryShader(regs);
}

bool RasterizerVulkan::AccelerateDrawBatch(bool is_indexed) {
    if (regs.pipeline.use_gs != Pica::PipelineRegs::UseGS::No) {
        if (regs.pipeline.gs_config.mode != Pica::PipelineRegs::GSMode::Point) {
            return false;
        }
        if (regs.pipeline.triangle_topology != Pica::PipelineRegs::TriangleTopology::Shader) {
            return false;
        }
    }

    pipeline_info.rasterization.topology.Assign(regs.pipeline.triangle_topology);
    if (regs.pipeline.triangle_topology == TriangleTopology::Fan &&
        !instance.IsTriangleFanSupported()) {
        LOG_DEBUG(Render_Vulkan,
                  "Skipping accelerated draw with unsupported triangle fan topology");
        return false;
    }

    // Vertex data setup might involve scheduler flushes so perform it
    // early to avoid invalidating our state in the middle of the draw.
    vertex_info = AnalyzeVertexArray(is_indexed, instance.GetMinVertexStrideAlignment());
    SetupVertexArray();

    if (!SetupVertexShader()) {
        return false;
    }
    if (!SetupGeometryShader()) {
        return false;
    }

    return Draw(true, is_indexed);
}

bool RasterizerVulkan::AccelerateDrawBatchInternal(bool is_indexed) {
    const bool tiny_textured_accelerated_draw_internal =
        IsStrictCompatEnabled() && is_indexed && regs.pipeline.num_vertices == 6 &&
        HasPrimaryTexturesEnabled(regs) && CountEnabledPrimaryTextures(regs) == 1 &&
        !regs.framebuffer.IsShadowRendering() && !HasActiveDepthState(regs);
    if (IsDrawTraceEnabled()) {
        LOG_INFO(Render_Vulkan,
                 "TRACE_DRAW accel_internal indexed={} vertex_count={} binding_count={}",
                 is_indexed, regs.pipeline.num_vertices, pipeline_info.vertex_layout.binding_count);
        if (tiny_textured_accelerated_draw_internal) {
            LOG_INFO(Render_Vulkan,
                     "TRACE_DRAW tiny_textured_accel_v5_before_internal vertex_count={} indexed={} binding_count={} patch_v5=1",
                     regs.pipeline.num_vertices, static_cast<u32>(is_indexed),
                     pipeline_info.vertex_layout.binding_count);
        }
    }

    if (regs.pipeline.num_vertices == 0) {
        if (IsDrawTraceEnabled()) {
            LOG_INFO(Render_Vulkan, "TRACE_DRAW accel_internal skipped empty draw");
        }
        return true;
    }

    const u32 binding_count = pipeline_info.vertex_layout.binding_count;
    if (binding_count == 0 || binding_count > vertex_buffers.size()) {
        LOG_ERROR(Render_Vulkan,
                  "Accelerated draw has invalid binding_count={} (max={})",
                  binding_count, vertex_buffers.size());
        return false;
    }

    if (is_indexed) {
        SetupIndexArray();
    }

    const bool wait_built = IsStrictCompatEnabled() ? true
                                                    : (!async_shaders || regs.pipeline.num_vertices <= 6);
    if (!pipeline_cache.BindPipeline(pipeline_info, wait_built)) {
        if (IsDrawTraceEnabled()) {
            LOG_INFO(Render_Vulkan,
                     "TRACE_DRAW pipeline not ready wait_built={} strict_compat={}",
                     wait_built, static_cast<u32>(IsStrictCompatEnabled()));
            if (tiny_textured_accelerated_draw_internal) {
                LOG_INFO(Render_Vulkan,
                         "TRACE_DRAW tiny_textured_accel_v5_pipeline_not_ready wait_built={} strict_compat={} patch_v5=1",
                         wait_built, static_cast<u32>(IsStrictCompatEnabled()));
            }
        }
        return false;
    }
    if (tiny_textured_accelerated_draw_internal && IsDrawTraceEnabled()) {
        LOG_INFO(Render_Vulkan,
                 "TRACE_DRAW tiny_textured_accel_v5_after_bind_pipeline wait_built={} patch_v5=1",
                 wait_built);
    }

    const DrawParams params = {
        .vertex_count = regs.pipeline.num_vertices,
        .vertex_offset = -static_cast<s32>(vertex_info.vs_input_index_min),
        .binding_count = binding_count,
        .bindings = binding_offsets,
        .is_indexed = is_indexed,
    };

    if (tiny_textured_accelerated_draw_internal && IsDrawTraceEnabled()) {
        LOG_INFO(Render_Vulkan,
                 "TRACE_DRAW tiny_textured_accel_v5_before_record vertex_count={} indexed={} patch_v5=1",
                 params.vertex_count, static_cast<u32>(params.is_indexed));
    }

    scheduler.Record([this, params](vk::CommandBuffer cmdbuf) {
        std::array<vk::DeviceSize, 16> offsets{};
        std::transform(params.bindings.begin(), params.bindings.end(), offsets.begin(),
                       [](u32 offset) { return static_cast<vk::DeviceSize>(offset); });
        cmdbuf.bindVertexBuffers(0, params.binding_count, vertex_buffers.data(), offsets.data());
        if (params.is_indexed) {
            cmdbuf.drawIndexed(params.vertex_count, 1, 0, params.vertex_offset, 0);
        } else {
            cmdbuf.draw(params.vertex_count, 1, 0, 0);
        }
    });

    if (tiny_textured_accelerated_draw_internal && IsDrawTraceEnabled()) {
        LOG_INFO(Render_Vulkan,
                 "TRACE_DRAW tiny_textured_accel_v5_after_record vertex_count={} indexed={} patch_v5=1",
                 params.vertex_count, static_cast<u32>(params.is_indexed));
        LOG_INFO(Render_Vulkan,
                 "TRACE_DRAW tiny_textured_accel_v5_after_internal vertex_count={} indexed={} patch_v5=1",
                 params.vertex_count, static_cast<u32>(params.is_indexed));
    }

    return true;
}

void RasterizerVulkan::SetupIndexArray() {
    const bool index_u8 = regs.pipeline.index_array.format == 0;
    const bool native_u8 = index_u8 && instance.IsIndexTypeUint8Supported();
    const u32 source_index_size = regs.pipeline.num_vertices * (index_u8 ? 1u : 2u);
    const u32 index_buffer_size = regs.pipeline.num_vertices * (native_u8 ? 1u : 2u);
    const vk::IndexType index_type = native_u8 ? vk::IndexType::eUint8EXT : vk::IndexType::eUint16;
    const PAddr index_addr =
        regs.pipeline.vertex_attributes.GetPhysicalBaseAddress() + regs.pipeline.index_array.offset;

    if (IsDrawTraceEnabled()) {
        LOG_INFO(Render_Vulkan,
                 "TRACE_DRAW setup_index_array addr=0x{:08x} num_vertices={} index_u8={} native_u8={} src_size={} dst_size={}",
                 index_addr, regs.pipeline.num_vertices, static_cast<u32>(index_u8),
                 static_cast<u32>(native_u8), source_index_size, index_buffer_size);
    }

    auto [index_ptr, index_offset, _] = stream_buffer.Map(index_buffer_size, 2);
    std::memset(index_ptr, 0, index_buffer_size);

    if (source_index_size != 0) {
        const MemoryRef index_ref = memory.GetPhysicalRef(index_addr);
        if (index_ref.GetSize() < source_index_size) {
            LOG_ERROR(Render_Vulkan,
                      "Index buffer size {} exceeds available space {} at address {:#016X}",
                      source_index_size, index_ref.GetSize(), index_addr);
        } else {
            const u8* index_data = index_ref.GetPtr();
            if (index_u8 && !native_u8) {
                u16* index_ptr_u16 = reinterpret_cast<u16*>(index_ptr);
                for (u32 i = 0; i < regs.pipeline.num_vertices; i++) {
                    index_ptr_u16[i] = index_data[i];
                }
            } else {
                std::memcpy(index_ptr, index_data, source_index_size);
            }
        }
    }

    stream_buffer.Commit(index_buffer_size);

    scheduler.Record(
        [this, index_offset = index_offset, index_type = index_type](vk::CommandBuffer cmdbuf) {
            cmdbuf.bindIndexBuffer(stream_buffer.Handle(), index_offset, index_type);
        });
}

void RasterizerVulkan::DrawTriangles() {
    LOG_DEBUG(Render_Vulkan, "Starting DrawTriangles with batch size {}", vertex_batch.size());

    if (vertex_batch.empty()) {
        LOG_DEBUG(Render_Vulkan, "Empty vertex batch, skipping draw");
        return;
    }

    try {
        pipeline_info.rasterization.topology.Assign(Pica::PipelineRegs::TriangleTopology::List);
        pipeline_info.vertex_layout = software_layout;

        pipeline_cache.UseTrivialVertexShader();
        pipeline_cache.UseTrivialGeometryShader();
        LOG_DEBUG(Render_Vulkan, "RasterizerVulkan::DrawTriangles pipeline_ready");
        if (IsDrawTraceEnabled()) {
            LOG_INFO(Render_Vulkan, "TRACE_DRAW draw_triangles software_batch_size={}", vertex_batch.size());
        }
        Draw(false, false);
        LOG_DEBUG(Render_Vulkan, "RasterizerVulkan::DrawTriangles draw_submitted");
    } catch (const vk::SystemError& e) {
        LOG_CRITICAL(Render_Vulkan, "Vulkan error in DrawTriangles: {}", e.what());
    } catch (const std::exception& e) {
        LOG_CRITICAL(Render_Vulkan, "Error in DrawTriangles: {}", e.what());
    }
}

bool RasterizerVulkan::Draw(bool accelerate, bool is_indexed) {

    BORKED3DS_PROFILE("Vulkan", "Drawing");
    if (IsDrawTraceEnabled()) {
        const u64 draw_index = ++g_vk_draw_counter;
        if (draw_index <= 16 || (draw_index % 256) == 0) {
            LOG_INFO(Render_Vulkan,
                     "TRACE_DRAW begin index={} accelerate={} indexed={} num_vertices={} topology={} shadow={} color_mask=0x{:x} depth_test={} depth_write={} stencil_test={}",
                     draw_index, accelerate, is_indexed, regs.pipeline.num_vertices,
                     static_cast<u32>(regs.pipeline.triangle_topology.Value()),
                     regs.framebuffer.IsShadowRendering(),
                     static_cast<u32>(pipeline_info.blending.color_write_mask),
                     static_cast<bool>(pipeline_info.depth_stencil.depth_test_enable),
                     static_cast<bool>(pipeline_info.depth_stencil.depth_write_enable),
                     static_cast<bool>(pipeline_info.depth_stencil.stencil_test_enable));
        }
    }
    const bool shadow_rendering = regs.framebuffer.IsShadowRendering();
    const bool has_stencil = regs.framebuffer.HasStencil();

    const bool write_color_fb = shadow_rendering || pipeline_info.blending.color_write_mask;
    const bool write_depth_fb = pipeline_info.IsDepthWriteEnabled();
    if (IsDrawTraceEnabled()) {
        LOG_INFO(Render_Vulkan,
                 "TRACE_DRAW targets write_color={} write_depth={} has_stencil={} color_addr=0x{:08x} depth_addr=0x{:08x}",
                 write_color_fb, write_depth_fb, has_stencil,
                 regs.framebuffer.framebuffer.GetColorBufferPhysicalAddress(),
                 regs.framebuffer.framebuffer.GetDepthBufferPhysicalAddress());
    }
    const bool using_color_fb =
        regs.framebuffer.framebuffer.GetColorBufferPhysicalAddress() != 0 && write_color_fb;
    const bool using_depth_fb =
        !shadow_rendering && regs.framebuffer.framebuffer.GetDepthBufferPhysicalAddress() != 0 &&
        (write_depth_fb || regs.framebuffer.output_merger.depth_test_enable != 0 ||
         (has_stencil && pipeline_info.depth_stencil.stencil_test_enable));

    const auto fb_helper = res_cache.GetFramebufferSurfaces(using_color_fb, using_depth_fb);
    const Framebuffer* framebuffer = fb_helper.Framebuffer();
    if (IsDrawTraceEnabled()) {
        LOG_INFO(Render_Vulkan,
                 "TRACE_DRAW framebuffer using_color={} using_depth={} fb_valid={} color_addr=0x{:08x} depth_addr=0x{:08x}",
                 using_color_fb, using_depth_fb, static_cast<bool>(framebuffer->Handle()),
                 regs.framebuffer.framebuffer.GetColorBufferPhysicalAddress(),
                 regs.framebuffer.framebuffer.GetDepthBufferPhysicalAddress());
    }
    if (!framebuffer->Handle()) {
        if (IsDrawTraceEnabled()) {
            LOG_INFO(Render_Vulkan, "TRACE_DRAW skipped: framebuffer handle invalid");
        }
        return true;
    }

    u64 startup_textured_software_skip_index = 0;
    const bool startup_textured_software_draw = [&] {
        if (accelerate || !ShouldSkipStartupTexturedSoftwareDraw(regs, vertex_batch.size())) {
            return false;
        }
        startup_textured_software_skip_index = ++g_vk_startup_textured_software_skip_counter;
        return startup_textured_software_skip_index <= 512;
    }();

    const bool tiny_textured_software_draw =
        !accelerate && ShouldAttemptTinyTexturedSoftwareDraw(regs, vertex_batch.size());
    const bool tiny_textured_accelerated_draw =
        accelerate && is_indexed && regs.pipeline.num_vertices == 6 &&
        HasPrimaryTexturesEnabled(regs) && CountEnabledPrimaryTextures(regs) == 1 &&
        !regs.framebuffer.IsShadowRendering() && !HasActiveDepthState(regs);
    u64 large_textured_software_draw_index = 0;
    const bool large_textured_software_draw = [&] {
        if (accelerate || !ShouldAttemptLargeTexturedSoftwareDraw(regs, vertex_batch.size())) {
            return false;
        }
        large_textured_software_draw_index = ++g_vk_large_textured_software_allow_counter;
        return large_textured_software_draw_index <= 320;
    }();

    u64 medium_textured_software_draw_index = 0;
    const bool medium_textured_software_draw = [&] {
        if (accelerate || !ShouldAttemptMediumTexturedSoftwareDraw(regs, vertex_batch.size())) {
            return false;
        }
        medium_textured_software_draw_index = ++g_vk_medium_textured_software_skip_counter;
        return medium_textured_software_draw_index <= 320;
    }();

    if (!accelerate) {
        if (ShouldSkipBatch42TexturedSoftwareDraw(regs, vertex_batch.size())) {
            const u64 batch42_skip_index = ++g_vk_batch42_textured_software_skip_counter;
            if (batch42_skip_index <= 256) {
                if (IsDrawTraceEnabled()) {
                    LOG_INFO(Render_Vulkan,
                             "TRACE_DRAW strict_compat early_skip_batch42_textured_software_draw_v13 skip_index={} vertex_batch_size={} num_vertices={} enabled_textures={} depth_active={} color_addr=0x{:08x} depth_addr=0x{:08x}",
                             batch42_skip_index, vertex_batch.size(), regs.pipeline.num_vertices,
                             CountEnabledPrimaryTextures(regs),
                             static_cast<u32>(HasActiveDepthState(regs)),
                             regs.framebuffer.framebuffer.GetColorBufferPhysicalAddress(),
                             regs.framebuffer.framebuffer.GetDepthBufferPhysicalAddress());
                }
                vertex_batch.clear();
                return true;
            }
        }

        if (ShouldSkipNonIndexed96TexturedSoftwareDraw(regs, vertex_batch.size())) {
            const u64 nonindexed96_skip_index = ++g_vk_nonindexed96_textured_software_skip_counter;
            if (nonindexed96_skip_index <= 64) {
                if (IsDrawTraceEnabled()) {
                    LOG_INFO(Render_Vulkan,
                             "TRACE_DRAW strict_compat early_skip_nonindexed96_textured_software_draw_v14 skip_index={} vertex_batch_size={} num_vertices={} enabled_textures={} depth_active={} color_addr=0x{:08x} depth_addr=0x{:08x}",
                             nonindexed96_skip_index, vertex_batch.size(), regs.pipeline.num_vertices,
                             CountEnabledPrimaryTextures(regs),
                             static_cast<u32>(HasActiveDepthState(regs)),
                             regs.framebuffer.framebuffer.GetColorBufferPhysicalAddress(),
                             regs.framebuffer.framebuffer.GetDepthBufferPhysicalAddress());
                }
                vertex_batch.clear();
                return true;
            }
        }

        if (ShouldSkipNonIndexed36TexturedSoftwareDraw(regs, vertex_batch.size())) {
            const u64 nonindexed36_skip_index = ++g_vk_nonindexed36_textured_software_skip_counter;
            if (nonindexed36_skip_index <= 64) {
                if (IsDrawTraceEnabled()) {
                    LOG_INFO(Render_Vulkan,
                             "TRACE_DRAW strict_compat early_skip_nonindexed36_textured_software_draw_v15 skip_index={} vertex_batch_size={} num_vertices={} enabled_textures={} depth_active={} color_addr=0x{:08x} depth_addr=0x{:08x}",
                             nonindexed36_skip_index, vertex_batch.size(), regs.pipeline.num_vertices,
                             CountEnabledPrimaryTextures(regs),
                             static_cast<u32>(HasActiveDepthState(regs)),
                             regs.framebuffer.framebuffer.GetColorBufferPhysicalAddress(),
                             regs.framebuffer.framebuffer.GetDepthBufferPhysicalAddress());
                }
                vertex_batch.clear();
                return true;
            }
        }

        if (startup_textured_software_draw) {
            if (IsDrawTraceEnabled()) {
                LOG_INFO(Render_Vulkan,
                         "TRACE_DRAW strict_compat early_skip_startup_textured_software_draw_v13 skip_index={} vertex_batch_size={} num_vertices={} enabled_textures={} depth_active={} color_addr=0x{:08x} depth_addr=0x{:08x}",
                         startup_textured_software_skip_index, vertex_batch.size(),
                         regs.pipeline.num_vertices, CountEnabledPrimaryTextures(regs),
                         static_cast<u32>(HasActiveDepthState(regs)),
                         regs.framebuffer.framebuffer.GetColorBufferPhysicalAddress(),
                         regs.framebuffer.framebuffer.GetDepthBufferPhysicalAddress());
            }
            vertex_batch.clear();
            return true;
        }

        if (ShouldBypassFragileSoftwareDraw(regs, vertex_batch.size())) {
            const u64 bypass_index = ++g_vk_software_bypass_counter;
            if (bypass_index <= 320) {
                if (IsDrawTraceEnabled()) {
                    LOG_INFO(Render_Vulkan,
                             "TRACE_DRAW strict_compat early_bypass_software_draw bypass_index={} vertex_batch_size={} num_vertices={} textures_disabled=1 color_addr=0x{:08x} depth_addr=0x{:08x}",
                             bypass_index, vertex_batch.size(), regs.pipeline.num_vertices,
                             regs.framebuffer.framebuffer.GetColorBufferPhysicalAddress(),
                             regs.framebuffer.framebuffer.GetDepthBufferPhysicalAddress());
                }
                vertex_batch.clear();
                return true;
            }
        }

        if (ShouldBypassFragileTexturedSoftwareDraw(regs, vertex_batch.size()) &&
            !large_textured_software_draw && !medium_textured_software_draw) {
            const u64 bypass_index = ++g_vk_textured_software_bypass_counter;
            if (bypass_index <= 320) {
                if (IsDrawTraceEnabled()) {
                    LOG_INFO(Render_Vulkan,
                             "TRACE_DRAW strict_compat early_bypass_textured_software_draw_v13 bypass_index={} vertex_batch_size={} num_vertices={} enabled_textures={} tiny_textured={} textures_disabled=0 depth_active={} color_addr=0x{:08x} depth_addr=0x{:08x}",
                             bypass_index, vertex_batch.size(), regs.pipeline.num_vertices,
                             CountEnabledPrimaryTextures(regs),
                             static_cast<u32>(tiny_textured_software_draw),
                             static_cast<u32>(HasActiveDepthState(regs)),
                             regs.framebuffer.framebuffer.GetColorBufferPhysicalAddress(),
                             regs.framebuffer.framebuffer.GetDepthBufferPhysicalAddress());
                }
                vertex_batch.clear();
                return true;
            }
        }

        if (tiny_textured_software_draw && IsDrawTraceEnabled()) {
            LOG_INFO(Render_Vulkan,
                     "TRACE_DRAW strict_compat allowing_tiny_textured_software_draw_after_extreme_bypass_window_v13 vertex_batch_size={} num_vertices={} enabled_textures={} depth_active={} color_addr=0x{:08x} depth_addr=0x{:08x}",
                     vertex_batch.size(), regs.pipeline.num_vertices,
                     CountEnabledPrimaryTextures(regs),
                     static_cast<u32>(HasActiveDepthState(regs)),
                     regs.framebuffer.framebuffer.GetColorBufferPhysicalAddress(),
                     regs.framebuffer.framebuffer.GetDepthBufferPhysicalAddress());
        }

        if (IsDrawTraceEnabled() && !large_textured_software_draw) {
            const u64 trace_index = ++g_vk_non_bypassed_software_trace_counter;
            if (trace_index <= 32) {
                LOG_INFO(Render_Vulkan,
                         "TRACE_DRAW software_draw_after_bypass_v3 trace_index={} vertex_batch_size={} num_vertices={} enabled_textures={} textures_disabled={} depth_active={} color_addr=0x{:08x} depth_addr=0x{:08x}",
                         trace_index, vertex_batch.size(), regs.pipeline.num_vertices,
                         CountEnabledPrimaryTextures(regs),
                         static_cast<u32>(ArePrimaryTexturesDisabled(regs)),
                         static_cast<u32>(HasActiveDepthState(regs)),
                         regs.framebuffer.framebuffer.GetColorBufferPhysicalAddress(),
                         regs.framebuffer.framebuffer.GetDepthBufferPhysicalAddress());
            }
        }

        if (medium_textured_software_draw) {
            if (IsDrawTraceEnabled()) {
                LOG_INFO(Render_Vulkan,
                         "TRACE_DRAW strict_compat early_skip_medium_textured_software_draw_v13 medium_index={} vertex_batch_size={} num_vertices={} enabled_textures={} depth_active={} color_addr=0x{:08x} depth_addr=0x{:08x}",
                         medium_textured_software_draw_index, vertex_batch.size(),
                         regs.pipeline.num_vertices, CountEnabledPrimaryTextures(regs),
                         static_cast<u32>(HasActiveDepthState(regs)),
                         regs.framebuffer.framebuffer.GetColorBufferPhysicalAddress(),
                         regs.framebuffer.framebuffer.GetDepthBufferPhysicalAddress());
            }
            vertex_batch.clear();
            return true;
        }

        if (large_textured_software_draw) {
            if (IsDrawTraceEnabled()) {
                LOG_INFO(Render_Vulkan,
                         "TRACE_DRAW strict_compat early_skip_large_textured_software_draw_v13 large_index={} vertex_batch_size={} num_vertices={} enabled_textures={} depth_active={} color_addr=0x{:08x} depth_addr=0x{:08x}",
                         large_textured_software_draw_index, vertex_batch.size(),
                         regs.pipeline.num_vertices, CountEnabledPrimaryTextures(regs),
                         static_cast<u32>(HasActiveDepthState(regs)),
                         regs.framebuffer.framebuffer.GetColorBufferPhysicalAddress(),
                         regs.framebuffer.framebuffer.GetDepthBufferPhysicalAddress());
            }
            vertex_batch.clear();
            return true;
        }
    }

    pipeline_info.attachments.color = framebuffer->Format(SurfaceType::Color);
    pipeline_info.attachments.depth = framebuffer->Format(SurfaceType::Depth);
    if (IsDrawTraceEnabled()) {
        LOG_INFO(Render_Vulkan,
                 "TRACE_DRAW attachments color_format={} depth_format={} using_color={} using_depth={}",
                 static_cast<u32>(pipeline_info.attachments.color),
                 static_cast<u32>(pipeline_info.attachments.depth), using_color_fb, using_depth_fb);
    }

    // Update scissor uniforms
    const auto [scissor_x1, scissor_y2, scissor_x2, scissor_y1] = fb_helper.Scissor();
    if (fs_uniform_block_data.data.scissor_x1 != scissor_x1 ||
        fs_uniform_block_data.data.scissor_x2 != scissor_x2 ||
        fs_uniform_block_data.data.scissor_y1 != scissor_y1 ||
        fs_uniform_block_data.data.scissor_y2 != scissor_y2) {

        fs_uniform_block_data.data.scissor_x1 = scissor_x1;
        fs_uniform_block_data.data.scissor_x2 = scissor_x2;
        fs_uniform_block_data.data.scissor_y1 = scissor_y1;
        fs_uniform_block_data.data.scissor_y2 = scissor_y2;
        fs_uniform_block_data.dirty = true;
    }

    // Sync and bind the texture surfaces
    if (large_textured_software_draw && IsDrawTraceEnabled()) {
        LOG_INFO(Render_Vulkan,
                 "TRACE_DRAW large_step_6_before_sync_textures large_index={} framebuffer_valid={} vertex_batch_size={}",
                 large_textured_software_draw_index, framebuffer != nullptr, vertex_batch.size());
    }
    SyncTextureUnits(framebuffer);
    SyncUtilityTextures(framebuffer);
    if (large_textured_software_draw && IsDrawTraceEnabled()) {
        LOG_INFO(Render_Vulkan,
                 "TRACE_DRAW large_step_7_after_sync_textures large_index={} enabled_textures={} vertex_batch_size={}",
                 large_textured_software_draw_index, CountEnabledPrimaryTextures(regs),
                 vertex_batch.size());
        LOG_INFO(Render_Vulkan,
                 "TRACE_DRAW large_step_8_before_fragment_shader large_index={} shader_dirty={}",
                 large_textured_software_draw_index, static_cast<u32>(shader_dirty));
    }

    if (shader_dirty) {
        Pica::Shader::UserConfig user_config{};
        const bool lighting_disabled = static_cast<bool>(regs.lighting.disable.Value());
        const bool use_custom_normal =
            (!lighting_disabled) && !instance.IsFragmentShaderBarycentricSupported();
        user_config.use_custom_normal.Assign(use_custom_normal);
        if (IsDrawTraceEnabled()) {
            LOG_INFO(Render_Vulkan,
                     "TRACE_DRAW use_fragment_shader shader_dirty=1 use_custom_normal={} lighting_disabled={} barycentric_supported={}",
                     static_cast<u32>(use_custom_normal),
                     static_cast<u32>(lighting_disabled),
                     static_cast<u32>(instance.IsFragmentShaderBarycentricSupported()));
        }
        pipeline_cache.UseFragmentShader(regs, user_config);
        shader_dirty = false;
    } else if (IsDrawTraceEnabled()) {
        LOG_INFO(Render_Vulkan, "TRACE_DRAW use_fragment_shader shader_dirty=0");
    }

    if (large_textured_software_draw && IsDrawTraceEnabled()) {
        LOG_INFO(Render_Vulkan,
                 "TRACE_DRAW large_step_9_after_fragment_shader large_index={} shader_dirty={}",
                 large_textured_software_draw_index, static_cast<u32>(shader_dirty));
        LOG_INFO(Render_Vulkan,
                 "TRACE_DRAW large_step_10_before_upload_uniforms large_index={} fs_dirty={} vs_dirty={} accelerate={}",
                 large_textured_software_draw_index,
                 static_cast<u32>(fs_uniform_block_data.dirty),
                 static_cast<u32>(vs_uniform_block_data.dirty), static_cast<u32>(accelerate));
    }

    // Sync the LUTs within the texture buffer
    SyncAndUploadLUTs();
    SyncAndUploadLUTsLF();
    UploadUniforms(accelerate);
    if (large_textured_software_draw && IsDrawTraceEnabled()) {
        LOG_INFO(Render_Vulkan,
                 "TRACE_DRAW large_step_11_after_upload_uniforms large_index={} fs_dirty={} vs_dirty={} accelerate={}",
                 large_textured_software_draw_index,
                 static_cast<u32>(fs_uniform_block_data.dirty),
                 static_cast<u32>(vs_uniform_block_data.dirty), static_cast<u32>(accelerate));
        LOG_INFO(Render_Vulkan,
                 "TRACE_DRAW large_step_12_before_flush large_index={} vertex_batch_size={}",
                 large_textured_software_draw_index, vertex_batch.size());
    }

    // Pi 5 / V3DV strict compatibility:
    // - make descriptor writes visible before pipeline binding / render begin
    // - serialize prior work before the first fragile software draw path
    update_queue.Flush();
    if (IsDrawTraceEnabled()) {
        LOG_INFO(Render_Vulkan, "TRACE_DRAW descriptors_flushed accelerate={}",
                 static_cast<u32>(accelerate));
    }
    if (large_textured_software_draw && IsDrawTraceEnabled()) {
        LOG_INFO(Render_Vulkan,
                 "TRACE_DRAW large_step_13_after_flush large_index={} accelerate={}",
                 large_textured_software_draw_index, static_cast<u32>(accelerate));
    }
    if (IsStrictCompatEnabled() && tiny_textured_accelerated_draw) {
        scheduler.Finish();
        if (IsDrawTraceEnabled()) {
            LOG_INFO(Render_Vulkan,
                     "TRACE_DRAW strict_compat serialized_before_tiny_textured_accel_draw_v4 vertex_count={} enabled_textures={} depth_active={} patch_v4=1",
                     regs.pipeline.num_vertices, CountEnabledPrimaryTextures(regs),
                     static_cast<u32>(HasActiveDepthState(regs)));
        }
    }
    if (IsStrictCompatEnabled() && !accelerate) {
        if (large_textured_software_draw) {
            if (IsDrawTraceEnabled()) {
                LOG_INFO(Render_Vulkan,
                         "TRACE_DRAW strict_compat skipping_finish_for_first_large_textured_software_draw_v2 large_index={} vertex_batch_size={} num_vertices={} enabled_textures={}",
                         large_textured_software_draw_index, vertex_batch.size(),
                         regs.pipeline.num_vertices, CountEnabledPrimaryTextures(regs));
                LOG_INFO(Render_Vulkan,
                         "TRACE_DRAW strict_compat serialized_before_software_draw vertex_batch_size={} finish_skipped={} patch_v2={}",
                         vertex_batch.size(), 1, 1);
                LOG_INFO(Render_Vulkan,
                         "TRACE_DRAW strict_compat serialized_before_large_textured_software_draw large_index={} vertex_batch_size={} num_vertices={} enabled_textures={} finish_skipped={} patch_v2={}",
                         large_textured_software_draw_index, vertex_batch.size(),
                         regs.pipeline.num_vertices, CountEnabledPrimaryTextures(regs), 1, 1);
                LOG_INFO(Render_Vulkan,
                         "TRACE_DRAW large_step_14_after_finish large_index={} vertex_batch_size={} num_vertices={} finish_skipped={} patch_v2={}",
                         large_textured_software_draw_index, vertex_batch.size(),
                         regs.pipeline.num_vertices, 1, 1);
            }
        } else {
            scheduler.Finish();
            if (IsDrawTraceEnabled()) {
                LOG_INFO(Render_Vulkan,
                         "TRACE_DRAW strict_compat serialized_before_software_draw vertex_batch_size={}",
                         vertex_batch.size());
                if (tiny_textured_software_draw) {
                    LOG_INFO(Render_Vulkan,
                             "TRACE_DRAW strict_compat serialized_before_tiny_textured_draw vertex_batch_size={} num_vertices={} enabled_textures={}",
                             vertex_batch.size(), regs.pipeline.num_vertices,
                             CountEnabledPrimaryTextures(regs));
                }
                if (large_textured_software_draw) {
                    LOG_INFO(Render_Vulkan,
                             "TRACE_DRAW strict_compat serialized_before_large_textured_software_draw large_index={} vertex_batch_size={} num_vertices={} enabled_textures={}",
                             large_textured_software_draw_index, vertex_batch.size(),
                             regs.pipeline.num_vertices, CountEnabledPrimaryTextures(regs));
                    LOG_INFO(Render_Vulkan,
                             "TRACE_DRAW large_step_14_after_finish large_index={} vertex_batch_size={} num_vertices={}",
                             large_textured_software_draw_index, vertex_batch.size(),
                             regs.pipeline.num_vertices);
                }
            }
        }
    }

    if (large_textured_software_draw) {
        if (IsDrawTraceEnabled()) {
            LOG_INFO(Render_Vulkan,
                     "TRACE_DRAW late_skip_first_large_textured_software_draw_v3 large_index={} vertex_batch_size={} num_vertices={} patch_v3=1",
                     large_textured_software_draw_index, vertex_batch.size(),
                     regs.pipeline.num_vertices);
        }
        vertex_batch.clear();
        return true;
    }

    // Begin rendering
    const auto draw_rect = fb_helper.DrawRect();
    if (tiny_textured_accelerated_draw && IsDrawTraceEnabled()) {
        LOG_INFO(Render_Vulkan,
                 "TRACE_DRAW tiny_textured_accel_v5_before_begin_rendering draw_rect=({}, {}, {}, {}) patch_v5=1",
                 draw_rect.left, draw_rect.bottom, draw_rect.right, draw_rect.top);
    }
    renderpass_cache.BeginRendering(framebuffer, draw_rect);
    if (tiny_textured_accelerated_draw && IsDrawTraceEnabled()) {
        LOG_INFO(Render_Vulkan,
                 "TRACE_DRAW tiny_textured_accel_begin_rendering_v4 draw_rect=({}, {}, {}, {}) patch_v4=1",
                 draw_rect.left, draw_rect.bottom, draw_rect.right, draw_rect.top);
        LOG_INFO(Render_Vulkan,
                 "TRACE_DRAW tiny_textured_accel_v5_after_begin_rendering draw_rect=({}, {}, {}, {}) patch_v5=1",
                 draw_rect.left, draw_rect.bottom, draw_rect.right, draw_rect.top);
    }
    if (large_textured_software_draw && IsDrawTraceEnabled()) {
        LOG_INFO(Render_Vulkan,
                 "TRACE_DRAW large_step_16_after_begin_rendering large_index={} draw_rect=({}, {}, {}, {})",
                 large_textured_software_draw_index, draw_rect.left, draw_rect.bottom,
                 draw_rect.right, draw_rect.top);
    }

    // Configure viewport and scissor
    const auto viewport = fb_helper.Viewport();
    if (IsDrawTraceEnabled()) {
        LOG_INFO(Render_Vulkan,
                 "TRACE_DRAW render_area x={} y={} w={} h={} viewport=({}, {}, {}, {})",
                 draw_rect.left, draw_rect.bottom, draw_rect.GetWidth(), draw_rect.GetHeight(),
                 viewport.x, viewport.y, viewport.width, viewport.height);
    }
    pipeline_info.dynamic.viewport = Common::Rectangle<s32>{
        viewport.x,
        viewport.y,
        viewport.x + viewport.width,
        viewport.y + viewport.height,
    };
    pipeline_info.dynamic.scissor = draw_rect;

    // Draw the vertex batch
    bool succeeded = true;
    if (accelerate) {
        if (tiny_textured_accelerated_draw && IsDrawTraceEnabled()) {
            LOG_INFO(Render_Vulkan,
                     "TRACE_DRAW tiny_textured_accel_before_submit_v4 vertex_count={} indexed={} patch_v4=1",
                     regs.pipeline.num_vertices, static_cast<u32>(is_indexed));
            LOG_INFO(Render_Vulkan,
                     "TRACE_DRAW tiny_textured_accel_v5_before_internal_call vertex_count={} indexed={} patch_v5=1",
                     regs.pipeline.num_vertices, static_cast<u32>(is_indexed));
        }
        succeeded = AccelerateDrawBatchInternal(is_indexed);
        if (tiny_textured_accelerated_draw && IsDrawTraceEnabled()) {
            LOG_INFO(Render_Vulkan,
                     "TRACE_DRAW tiny_textured_accel_after_submit_v4 succeeded={} vertex_count={} patch_v4=1",
                     static_cast<u32>(succeeded), regs.pipeline.num_vertices);
            LOG_INFO(Render_Vulkan,
                     "TRACE_DRAW tiny_textured_accel_v5_after_internal_call succeeded={} vertex_count={} patch_v5=1",
                     static_cast<u32>(succeeded), regs.pipeline.num_vertices);
        }
    } else {
        if (IsDrawTraceEnabled()) {
            LOG_INFO(Render_Vulkan, "TRACE_DRAW software_path vertex_batch_size={}", vertex_batch.size());
        }
        if (large_textured_software_draw && IsDrawTraceEnabled()) {
            LOG_INFO(Render_Vulkan,
                     "TRACE_DRAW large_step_17_before_bind_pipeline large_index={} vertex_batch_size={} num_vertices={}",
                     large_textured_software_draw_index, vertex_batch.size(),
                     regs.pipeline.num_vertices);
        }

        const bool pipeline_ready = pipeline_cache.BindPipeline(pipeline_info, true);
        if (!pipeline_ready) {
            if (IsDrawTraceEnabled()) {
                LOG_INFO(Render_Vulkan,
                         "TRACE_DRAW software_path pipeline_not_ready vertex_batch_size={} strict_compat={}",
                         vertex_batch.size(), static_cast<u32>(IsStrictCompatEnabled()));
                if (large_textured_software_draw) {
                    LOG_INFO(Render_Vulkan,
                             "TRACE_DRAW large_textured_software_draw_failed large_index={} vertex_batch_size={} num_vertices={} reason=pipeline_not_ready",
                             large_textured_software_draw_index, vertex_batch.size(),
                             regs.pipeline.num_vertices);
                }
            }
            return false;
        }
        if (large_textured_software_draw && IsDrawTraceEnabled()) {
            LOG_INFO(Render_Vulkan,
                     "TRACE_DRAW large_step_18_after_bind_pipeline large_index={} vertex_batch_size={} num_vertices={}",
                     large_textured_software_draw_index, vertex_batch.size(),
                     regs.pipeline.num_vertices);
        }

        const u32 vertex_count = static_cast<u32>(vertex_batch.size());
        if (vertex_count == 0) {
            if (IsDrawTraceEnabled()) {
                LOG_INFO(Render_Vulkan, "TRACE_DRAW software_path skipped empty vertex batch");
            }
            return true;
        }

        const u32 vertex_size = vertex_count * sizeof(HardwareVertex);
        const auto [buffer, offset, _] = stream_buffer.Map(vertex_size, sizeof(HardwareVertex));

        std::memcpy(buffer, vertex_batch.data(), vertex_size);
        stream_buffer.Commit(vertex_size);

        scheduler.Record([this, offset = offset, vertex_count](vk::CommandBuffer cmdbuf) {
            cmdbuf.bindVertexBuffers(0, stream_buffer.Handle(), offset);
            cmdbuf.draw(vertex_count, 1, 0, 0);
        });

        if (IsDrawTraceEnabled()) {
            LOG_INFO(Render_Vulkan,
                     "TRACE_DRAW software_path submitted vertex_count={} buffer_offset={}",
                     vertex_count, offset);
            if (tiny_textured_software_draw) {
                LOG_INFO(Render_Vulkan,
                         "TRACE_DRAW tiny_textured_draw_submitted vertex_count={} buffer_offset={} enabled_textures={} depth_active={}",
                         vertex_count, offset, CountEnabledPrimaryTextures(regs),
                         static_cast<u32>(HasActiveDepthState(regs)));
            }
            if (large_textured_software_draw) {
                LOG_INFO(Render_Vulkan,
                         "TRACE_DRAW large_textured_software_draw_submitted large_index={} vertex_count={} buffer_offset={} enabled_textures={} depth_active={}",
                         large_textured_software_draw_index, vertex_count, offset,
                         CountEnabledPrimaryTextures(regs),
                         static_cast<u32>(HasActiveDepthState(regs)));
            }
        }
    }

    if (IsDrawTraceEnabled()) {
        LOG_INFO(Render_Vulkan, "TRACE_DRAW end succeeded={} remaining_batch={}", succeeded, vertex_batch.size());
    }
    vertex_batch.clear();
    return succeeded;
}

void RasterizerVulkan::SyncTextureUnits(const Framebuffer* framebuffer) {
    if (IsDrawTraceEnabled()) {
        LOG_INFO(Render_Vulkan, "TRACE_DRAW sync_textures begin framebuffer_valid={}", framebuffer != nullptr);
    }
    using TextureType = Pica::TexturingRegs::TextureConfig::TextureType;

    const auto pica_textures = regs.texturing.GetTextures();
    const bool use_cube_heap =
        pica_textures[0].enabled && pica_textures[0].config.type == TextureType::ShadowCube;
    const auto texture_set = pipeline_cache.Acquire(use_cube_heap ? DescriptorHeapType::Texture
                                                                  : DescriptorHeapType::Texture);

    const Surface& null_surface = res_cache.GetSurface(VideoCore::NULL_SURFACE_ID);
    const Sampler& null_sampler = res_cache.GetSampler(VideoCore::NULL_SAMPLER_ID);
    const vk::ImageView null_view = null_surface.ImageView();
    const vk::Sampler null_handle = null_sampler.Handle();
    const vk::ImageView color_view =
        framebuffer ? framebuffer->ImageView(SurfaceType::Color) : vk::ImageView{};

    for (u32 texture_index = 0; texture_index < pica_textures.size(); ++texture_index) {
        const auto& texture = pica_textures[texture_index];

        auto bind_null = [&](const char* reason) {
            if (IsDrawTraceEnabled() && texture_index < 3) {
                LOG_INFO(Render_Vulkan,
                         "TRACE_DRAW tex{} -> null reason={} type={} format={}",
                         texture_index, reason,
                         static_cast<u32>(texture.config.type.Value()),
                         static_cast<u32>(texture.format));
            }
            update_queue.AddImageSampler(texture_set, texture_index, 0, null_view, null_handle);
        };

        // If the texture unit is disabled bind a null surface to it
        if (!texture.enabled) {
            bind_null("disabled");
            continue;
        }

        // Handle special tex0 configurations
        if (texture_index == 0) {
            switch (texture.config.type.Value()) {
            case TextureType::Shadow2D: {
                Surface& surface = res_cache.GetTextureSurface(texture);
                Sampler& sampler = res_cache.GetSampler(texture.config);
                surface.flags |= VideoCore::SurfaceFlagBits::ShadowMap;
                const vk::ImageView view = surface.ImageView();
                if (!IsValidImageView(view)) {
                    bind_null("shadow2d_invalid_view");
                } else {
                    if (IsDrawTraceEnabled() && texture_index < 3) {
                        LOG_INFO(Render_Vulkan,
                                 "TRACE_DRAW tex{} shadow2d bound sampler_valid={}",
                                 texture_index, static_cast<bool>(sampler.Handle()));
                    }
                    update_queue.AddImageSampler(texture_set, texture_index, 0, view,
                                                 sampler.Handle());
                }
                continue;
            }
            case TextureType::ShadowCube: {
                BindShadowCube(texture, texture_set);
                continue;
            }
            case TextureType::TextureCube: {
                BindTextureCube(texture, texture_set);
                continue;
            }
            default:
                break;
            }
        }

        // Bind the texture provided by the rasterizer cache.
        // Pi 5 / V3DV stability fix:
        // - never submit a null ImageView
        // - avoid direct feedback loops by preferring CopyImageView()
        Surface& surface = res_cache.GetTextureSurface(texture);
        Sampler& sampler = res_cache.GetSampler(texture.config);

        const vk::ImageView base_view = surface.ImageView();
        const vk::ImageView copy_view = surface.CopyImageView();

        if (!IsValidImageView(base_view) && !IsValidImageView(copy_view)) {
            bind_null("invalid_base_and_copy_view");
            continue;
        }

        const bool strict_compat = IsStrictCompatEnabled();
        const bool direct_feedback = IsValidImageView(color_view) && color_view == base_view;

        vk::ImageView texture_view = base_view;
        const char* bind_reason = "base_view";

        if (strict_compat && IsValidImageView(copy_view)) {
            texture_view = copy_view;
            bind_reason = "strict_compat_copy";
        } else if (direct_feedback) {
            if (IsValidImageView(copy_view)) {
                texture_view = copy_view;
                bind_reason = "feedback_copy";
            } else if (!IsValidImageView(base_view)) {
                bind_null("feedback_copy_missing");
                continue;
            }
        } else if (!IsValidImageView(base_view) && IsValidImageView(copy_view)) {
            texture_view = copy_view;
            bind_reason = "copy_fallback";
        }

        if (IsDrawTraceEnabled() && texture_index < 3) {
            LOG_INFO(Render_Vulkan,
                     "TRACE_DRAW tex{} bound reason={} sampler_valid={} type={} format={} strict_compat={} direct_feedback={}",
                     texture_index, bind_reason, static_cast<bool>(sampler.Handle()),
                     static_cast<u32>(texture.config.type.Value()),
                     static_cast<u32>(texture.format), static_cast<u32>(strict_compat),
                     static_cast<u32>(direct_feedback));
        }
        update_queue.AddImageSampler(texture_set, texture_index, 0, texture_view,
                                     sampler.Handle());
    }
}

void RasterizerVulkan::SyncUtilityTextures(const Framebuffer* framebuffer) {
    const bool shadow_rendering = regs.framebuffer.IsShadowRendering();
    if (!shadow_rendering) {
        return;
    }

    const auto utility_set = pipeline_cache.Acquire(DescriptorHeapType::Utility);
    update_queue.AddStorageImage(utility_set, 0, framebuffer->ImageView(SurfaceType::Color));
}

void RasterizerVulkan::BindShadowCube(const Pica::TexturingRegs::FullTextureConfig& texture,
                                      vk::DescriptorSet texture_set) {
    using CubeFace = Pica::TexturingRegs::CubeFace;
    auto info = Pica::Texture::TextureInfo::FromPicaRegister(texture.config, texture.format);
    constexpr std::array faces = {
        CubeFace::PositiveX, CubeFace::NegativeX, CubeFace::PositiveY,
        CubeFace::NegativeY, CubeFace::PositiveZ, CubeFace::NegativeZ,
    };

    Sampler& sampler = res_cache.GetSampler(texture.config);

    for (CubeFace face : faces) {
        const u32 binding = static_cast<u32>(face);
        info.physical_address = regs.texturing.GetCubePhysicalAddress(face);

        const VideoCore::SurfaceId surface_id = res_cache.GetTextureSurface(info);
        Surface& surface = res_cache.GetSurface(surface_id);
        surface.flags |= VideoCore::SurfaceFlagBits::ShadowMap;
        update_queue.AddImageSampler(texture_set, 0, binding, surface.ImageView(),
                                     sampler.Handle());
    }
}

void RasterizerVulkan::BindTextureCube(const Pica::TexturingRegs::FullTextureConfig& texture,
                                       vk::DescriptorSet texture_set) {
    using CubeFace = Pica::TexturingRegs::CubeFace;
    const VideoCore::TextureCubeConfig config = {
        .px = regs.texturing.GetCubePhysicalAddress(CubeFace::PositiveX),
        .nx = regs.texturing.GetCubePhysicalAddress(CubeFace::NegativeX),
        .py = regs.texturing.GetCubePhysicalAddress(CubeFace::PositiveY),
        .ny = regs.texturing.GetCubePhysicalAddress(CubeFace::NegativeY),
        .pz = regs.texturing.GetCubePhysicalAddress(CubeFace::PositiveZ),
        .nz = regs.texturing.GetCubePhysicalAddress(CubeFace::NegativeZ),
        .width = texture.config.width,
        .levels = texture.config.lod.max_level + 1,
        .format = texture.format,
    };

    Surface& surface = res_cache.GetTextureCube(config);
    Sampler& sampler = res_cache.GetSampler(texture.config);
    update_queue.AddImageSampler(texture_set, 0, 0, surface.ImageView(), sampler.Handle());
}

void RasterizerVulkan::NotifyFixedFunctionPicaRegisterChanged(u32 id) {
    switch (id) {
    // Culling
    case PICA_REG_INDEX(rasterizer.cull_mode):
        SyncCullMode();
        break;

    // Blending
    case PICA_REG_INDEX(framebuffer.output_merger.alphablend_enable):
        SyncBlendEnabled();
        // Update since logic op emulation depends on alpha blend enable.
        SyncLogicOp();
        SyncColorWriteMask();
        break;
    case PICA_REG_INDEX(framebuffer.output_merger.alpha_blending):
        SyncBlendFuncs();
        break;
    case PICA_REG_INDEX(framebuffer.output_merger.blend_const):
        SyncBlendColor();
        break;

    // Sync VK stencil test + stencil write mask
    // (Pica stencil test function register also contains a stencil write mask)
    case PICA_REG_INDEX(framebuffer.output_merger.stencil_test.raw_func):
        SyncStencilTest();
        SyncStencilWriteMask();
        break;
    case PICA_REG_INDEX(framebuffer.output_merger.stencil_test.raw_op):
    case PICA_REG_INDEX(framebuffer.framebuffer.depth_format):
        SyncStencilTest();
        break;

    // Sync VK depth test + depth and color write mask
    // (Pica depth test function register also contains a depth and color write mask)
    case PICA_REG_INDEX(framebuffer.output_merger.depth_test_enable):
        SyncDepthTest();
        SyncDepthWriteMask();
        SyncColorWriteMask();
        break;

    // Sync VK depth and stencil write mask
    // (This is a dedicated combined depth / stencil write-enable register)
    case PICA_REG_INDEX(framebuffer.framebuffer.allow_depth_stencil_write):
        SyncDepthWriteMask();
        SyncStencilWriteMask();
        break;

    // Sync VK color write mask
    // (This is a dedicated color write-enable register)
    case PICA_REG_INDEX(framebuffer.framebuffer.allow_color_write):
        SyncColorWriteMask();
        break;

    // Logic op
    case PICA_REG_INDEX(framebuffer.output_merger.logic_op):
        SyncLogicOp();
        // Update since color write mask is used to emulate no-op.
        SyncColorWriteMask();
        break;
    }
}

void RasterizerVulkan::FlushAll() {
    res_cache.FlushAll();
}

void RasterizerVulkan::FlushRegion(PAddr addr, u32 size) {
    res_cache.FlushRegion(addr, size);
}

void RasterizerVulkan::InvalidateRegion(PAddr addr, u32 size) {
    res_cache.InvalidateRegion(addr, size);
}

void RasterizerVulkan::FlushAndInvalidateRegion(PAddr addr, u32 size) {
    res_cache.FlushRegion(addr, size);
    res_cache.InvalidateRegion(addr, size);
}

void RasterizerVulkan::ClearAll(bool flush) {
    res_cache.ClearAll(flush);
}

bool RasterizerVulkan::AccelerateDisplayTransfer(const Pica::DisplayTransferConfig& config) {
    return res_cache.AccelerateDisplayTransfer(config);
}

bool RasterizerVulkan::AccelerateTextureCopy(const Pica::DisplayTransferConfig& config) {
    return res_cache.AccelerateTextureCopy(config);
}

bool RasterizerVulkan::AccelerateFill(const Pica::MemoryFillConfig& config) {
    return res_cache.AccelerateFill(config);
}

bool RasterizerVulkan::AccelerateDisplay(const Pica::FramebufferConfig& config,
                                         PAddr framebuffer_addr, u32 pixel_stride,
                                         ScreenInfo& screen_info) {
    if (framebuffer_addr == 0) [[unlikely]] {
        return false;
    }

    VideoCore::SurfaceParams src_params;
    src_params.addr = framebuffer_addr;
    src_params.width = std::min(config.width.Value(), pixel_stride);
    src_params.height = config.height;
    src_params.stride = pixel_stride;
    src_params.is_tiled = false;
    src_params.pixel_format = VideoCore::PixelFormatFromGPUPixelFormat(config.color_format);
    src_params.UpdateParams();

    const auto [src_surface_id, src_rect] =
        res_cache.GetSurfaceSubRect(src_params, VideoCore::ScaleMatch::Ignore, true);

    if (!src_surface_id) {
        return false;
    }

    Surface& src_surface = res_cache.GetSurface(src_surface_id);
    const u32 scaled_width = src_surface.GetScaledWidth();
    const u32 scaled_height = src_surface.GetScaledHeight();

    screen_info.texcoords = Common::Rectangle<f32>(
        (float)src_rect.bottom / (float)scaled_height, (float)src_rect.left / (float)scaled_width,
        (float)src_rect.top / (float)scaled_height, (float)src_rect.right / (float)scaled_width);

    const vk::ImageView base_view = src_surface.ImageView();
    const vk::ImageView copy_view = src_surface.CopyImageView();
    const bool strict_compat = IsStrictCompatEnabled();
    screen_info.image_view = (strict_compat && IsValidImageView(copy_view)) ? copy_view : base_view;

    if (IsDrawTraceEnabled()) {
        LOG_INFO(Render_Vulkan,
                 "TRACE_DRAW accelerate_display addr=0x{:08x} width={} height={} stride={} pixel_format={} src_rect=({}, {}, {}, {}) base_valid={} copy_valid={} chosen={} strict_compat={}",
                 framebuffer_addr, src_params.width, src_params.height, src_params.stride,
                 static_cast<u32>(src_params.pixel_format), src_rect.left, src_rect.bottom,
                 src_rect.right, src_rect.top, static_cast<u32>(static_cast<bool>(base_view)),
                 static_cast<u32>(static_cast<bool>(copy_view)),
                 static_cast<u32>(static_cast<bool>(screen_info.image_view)),
                 static_cast<u32>(strict_compat));
    }

    return static_cast<bool>(screen_info.image_view);
}

void RasterizerVulkan::MakeSoftwareVertexLayout() {
    constexpr std::array sizes = {4, 4, 2, 2, 2, 1, 4, 3};

    software_layout = VertexLayout{
        .binding_count = 1,
        .attribute_count = 8,
    };

    for (u32 i = 0; i < software_layout.binding_count; i++) {
        VertexBinding& binding = software_layout.bindings[i];
        binding.binding.Assign(i);
        binding.fixed.Assign(0);
        binding.stride.Assign(sizeof(HardwareVertex));
    }

    u32 offset = 0;
    for (u32 i = 0; i < 8; i++) {
        VertexAttribute& attribute = software_layout.attributes[i];
        attribute.binding.Assign(0);
        attribute.location.Assign(i);
        attribute.offset.Assign(offset);
        attribute.type.Assign(Pica::PipelineRegs::VertexAttributeFormat::FLOAT);
        attribute.size.Assign(sizes[i]);
        offset += sizes[i] * sizeof(float);
    }
}

void RasterizerVulkan::SyncCullMode() {
    pipeline_info.rasterization.cull_mode.Assign(regs.rasterizer.cull_mode);
}

void RasterizerVulkan::SyncBlendEnabled() {
    pipeline_info.blending.blend_enable = regs.framebuffer.output_merger.alphablend_enable;
}

void RasterizerVulkan::SyncBlendFuncs() {
    pipeline_info.blending.color_blend_eq.Assign(
        regs.framebuffer.output_merger.alpha_blending.blend_equation_rgb);
    pipeline_info.blending.alpha_blend_eq.Assign(
        regs.framebuffer.output_merger.alpha_blending.blend_equation_a);
    pipeline_info.blending.src_color_blend_factor.Assign(
        regs.framebuffer.output_merger.alpha_blending.factor_source_rgb);
    pipeline_info.blending.dst_color_blend_factor.Assign(
        regs.framebuffer.output_merger.alpha_blending.factor_dest_rgb);
    pipeline_info.blending.src_alpha_blend_factor.Assign(
        regs.framebuffer.output_merger.alpha_blending.factor_source_a);
    pipeline_info.blending.dst_alpha_blend_factor.Assign(
        regs.framebuffer.output_merger.alpha_blending.factor_dest_a);
}

void RasterizerVulkan::SyncBlendColor() {
    pipeline_info.dynamic.blend_color = regs.framebuffer.output_merger.blend_const.raw;
}

void RasterizerVulkan::SyncLogicOp() {
    if (instance.NeedsLogicOpEmulation()) {
        // We need this in the fragment shader to emulate logic operations
        shader_dirty = true;
    }

    pipeline_info.blending.logic_op = regs.framebuffer.output_merger.logic_op;

    const bool is_logic_op_emulated =
        instance.NeedsLogicOpEmulation() && !regs.framebuffer.output_merger.alphablend_enable;
    const bool is_logic_op_noop =
        regs.framebuffer.output_merger.logic_op == Pica::FramebufferRegs::LogicOp::NoOp;
    if (is_logic_op_emulated && is_logic_op_noop) {
        // Color output is disabled by logic operation. We use color write mask to skip
        // color but allow depth write.
        pipeline_info.blending.color_write_mask = 0;
    }
}

void RasterizerVulkan::SyncColorWriteMask() {
    const u32 color_mask = regs.framebuffer.framebuffer.allow_color_write != 0
                               ? (regs.framebuffer.output_merger.depth_color_mask >> 8) & 0xF
                               : 0;

    const bool is_logic_op_emulated =
        instance.NeedsLogicOpEmulation() && !regs.framebuffer.output_merger.alphablend_enable;
    const bool is_logic_op_noop =
        regs.framebuffer.output_merger.logic_op == Pica::FramebufferRegs::LogicOp::NoOp;
    if (is_logic_op_emulated && is_logic_op_noop) {
        // Color output is disabled by logic operation. We use color write mask to skip
        // color but allow depth write. Return early to avoid overwriting this.
        return;
    }

    pipeline_info.blending.color_write_mask = color_mask;
}

void RasterizerVulkan::SyncStencilWriteMask() {
    pipeline_info.dynamic.stencil_write_mask =
        (regs.framebuffer.framebuffer.allow_depth_stencil_write != 0)
            ? static_cast<u32>(regs.framebuffer.output_merger.stencil_test.write_mask)
            : 0;
}

void RasterizerVulkan::SyncDepthWriteMask() {
    const bool write_enable = (regs.framebuffer.framebuffer.allow_depth_stencil_write != 0 &&
                               regs.framebuffer.output_merger.depth_write_enable);
    pipeline_info.depth_stencil.depth_write_enable.Assign(write_enable);
}

void RasterizerVulkan::SyncStencilTest() {
    const auto& stencil_test = regs.framebuffer.output_merger.stencil_test;
    const bool test_enable = stencil_test.enable && regs.framebuffer.framebuffer.depth_format ==
                                                        Pica::FramebufferRegs::DepthFormat::D24S8;

    pipeline_info.depth_stencil.stencil_test_enable.Assign(test_enable);
    pipeline_info.depth_stencil.stencil_fail_op.Assign(stencil_test.action_stencil_fail);
    pipeline_info.depth_stencil.stencil_pass_op.Assign(stencil_test.action_depth_pass);
    pipeline_info.depth_stencil.stencil_depth_fail_op.Assign(stencil_test.action_depth_fail);
    pipeline_info.depth_stencil.stencil_compare_op.Assign(stencil_test.func);
    pipeline_info.dynamic.stencil_reference = stencil_test.reference_value;
    pipeline_info.dynamic.stencil_compare_mask = stencil_test.input_mask;
}

void RasterizerVulkan::SyncDepthTest() {
    const bool test_enabled = regs.framebuffer.output_merger.depth_test_enable == 1 ||
                              regs.framebuffer.output_merger.depth_write_enable == 1;
    const auto compare_op = regs.framebuffer.output_merger.depth_test_enable == 1
                                ? regs.framebuffer.output_merger.depth_test_func.Value()
                                : Pica::FramebufferRegs::CompareFunc::Always;

    pipeline_info.depth_stencil.depth_test_enable.Assign(test_enabled);
    pipeline_info.depth_stencil.depth_compare_op.Assign(compare_op);
}

void RasterizerVulkan::SyncAndUploadLUTsLF() {
    constexpr std::size_t max_size =
        sizeof(Common::Vec2f) * 256 * Pica::LightingRegs::NumLightingSampler +
        sizeof(Common::Vec2f) * 128; // fog

    if (!fs_uniform_block_data.lighting_lut_dirty_any && !fs_uniform_block_data.fog_lut_dirty) {
        return;
    }

    std::size_t bytes_used = 0;
    auto [buffer, offset, invalidate] = texture_lf_buffer.Map(max_size, sizeof(Common::Vec4f));

    // Sync the lighting luts
    if (fs_uniform_block_data.lighting_lut_dirty_any || invalidate) {
        for (unsigned index = 0; index < fs_uniform_block_data.lighting_lut_dirty.size(); index++) {
            if (fs_uniform_block_data.lighting_lut_dirty[index] || invalidate) {
                std::array<Common::Vec2f, 256> new_data;
                const auto& source_lut = pica.lighting.luts[index];
                std::transform(source_lut.begin(), source_lut.end(), new_data.begin(),
                               [](const auto& entry) {
                                   return Common::Vec2f{entry.ToFloat(), entry.DiffToFloat()};
                               });

                if (new_data != lighting_lut_data[index] || invalidate) {
                    lighting_lut_data[index] = new_data;
                    std::memcpy(buffer + bytes_used, new_data.data(),
                                new_data.size() * sizeof(Common::Vec2f));
                    fs_uniform_block_data.data.lighting_lut_offset[index / 4][index % 4] =
                        static_cast<int>((offset + bytes_used) / sizeof(Common::Vec2f));
                    fs_uniform_block_data.dirty = true;
                    bytes_used += new_data.size() * sizeof(Common::Vec2f);
                }
                fs_uniform_block_data.lighting_lut_dirty[index] = false;
            }
        }
        fs_uniform_block_data.lighting_lut_dirty_any = false;
    }

    // Sync the fog lut
    if (fs_uniform_block_data.fog_lut_dirty || invalidate) {
        std::array<Common::Vec2f, 128> new_data;

        std::transform(
            pica.fog.lut.begin(), pica.fog.lut.end(), new_data.begin(),
            [](const auto& entry) { return Common::Vec2f{entry.ToFloat(), entry.DiffToFloat()}; });

        if (new_data != fog_lut_data || invalidate) {
            fog_lut_data = new_data;
            std::memcpy(buffer + bytes_used, new_data.data(),
                        new_data.size() * sizeof(Common::Vec2f));
            fs_uniform_block_data.data.fog_lut_offset =
                static_cast<int>((offset + bytes_used) / sizeof(Common::Vec2f));
            fs_uniform_block_data.dirty = true;
            bytes_used += new_data.size() * sizeof(Common::Vec2f);
        }
        fs_uniform_block_data.fog_lut_dirty = false;
    }

    texture_lf_buffer.Commit(static_cast<u32>(bytes_used));
}

void RasterizerVulkan::SyncAndUploadLUTs() {
    const auto& proctex = pica.proctex;
    constexpr std::size_t max_size =
        sizeof(Common::Vec2f) * 128 * 3 + // proctex: noise + color + alpha
        sizeof(Common::Vec4f) * 256 +     // proctex
        sizeof(Common::Vec4f) * 256;      // proctex diff

    if (!fs_uniform_block_data.proctex_noise_lut_dirty &&
        !fs_uniform_block_data.proctex_color_map_dirty &&
        !fs_uniform_block_data.proctex_alpha_map_dirty &&
        !fs_uniform_block_data.proctex_lut_dirty && !fs_uniform_block_data.proctex_diff_lut_dirty) {
        return;
    }

    std::size_t bytes_used = 0;
    auto [buffer, offset, invalidate] = texture_buffer.Map(max_size, sizeof(Common::Vec4f));

    // helper function for SyncProcTexNoiseLUT/ColorMap/AlphaMap
    auto sync_proctex_value_lut =
        [this, buffer = buffer, offset = offset, invalidate = invalidate,
         &bytes_used](const std::array<Pica::PicaCore::ProcTex::ValueEntry, 128>& lut,
                      std::array<Common::Vec2f, 128>& lut_data, int& lut_offset) {
            std::array<Common::Vec2f, 128> new_data;
            std::transform(lut.begin(), lut.end(), new_data.begin(), [](const auto& entry) {
                return Common::Vec2f{entry.ToFloat(), entry.DiffToFloat()};
            });

            if (new_data != lut_data || invalidate) {
                lut_data = new_data;
                std::memcpy(buffer + bytes_used, new_data.data(),
                            new_data.size() * sizeof(Common::Vec2f));
                lut_offset = static_cast<int>((offset + bytes_used) / sizeof(Common::Vec2f));
                fs_uniform_block_data.dirty = true;
                bytes_used += new_data.size() * sizeof(Common::Vec2f);
            }
        };

    // Sync the proctex noise lut
    if (fs_uniform_block_data.proctex_noise_lut_dirty || invalidate) {
        sync_proctex_value_lut(proctex.noise_table, proctex_noise_lut_data,
                               fs_uniform_block_data.data.proctex_noise_lut_offset);
        fs_uniform_block_data.proctex_noise_lut_dirty = false;
    }

    // Sync the proctex color map
    if (fs_uniform_block_data.proctex_color_map_dirty || invalidate) {
        sync_proctex_value_lut(proctex.color_map_table, proctex_color_map_data,
                               fs_uniform_block_data.data.proctex_color_map_offset);
        fs_uniform_block_data.proctex_color_map_dirty = false;
    }

    // Sync the proctex alpha map
    if (fs_uniform_block_data.proctex_alpha_map_dirty || invalidate) {
        sync_proctex_value_lut(proctex.alpha_map_table, proctex_alpha_map_data,
                               fs_uniform_block_data.data.proctex_alpha_map_offset);
        fs_uniform_block_data.proctex_alpha_map_dirty = false;
    }

    // Sync the proctex lut
    if (fs_uniform_block_data.proctex_lut_dirty || invalidate) {
        std::array<Common::Vec4f, 256> new_data;

        std::transform(proctex.color_table.begin(), proctex.color_table.end(), new_data.begin(),
                       [](const auto& entry) {
                           auto rgba = entry.ToVector() / 255.0f;
                           return Common::Vec4f{rgba.r(), rgba.g(), rgba.b(), rgba.a()};
                       });

        if (new_data != proctex_lut_data || invalidate) {
            proctex_lut_data = new_data;
            std::memcpy(buffer + bytes_used, new_data.data(),
                        new_data.size() * sizeof(Common::Vec4f));
            fs_uniform_block_data.data.proctex_lut_offset =
                static_cast<int>((offset + bytes_used) / sizeof(Common::Vec4f));
            fs_uniform_block_data.dirty = true;
            bytes_used += new_data.size() * sizeof(Common::Vec4f);
        }
        fs_uniform_block_data.proctex_lut_dirty = false;
    }

    // Sync the proctex difference lut
    if (fs_uniform_block_data.proctex_diff_lut_dirty || invalidate) {
        std::array<Common::Vec4f, 256> new_data;

        std::transform(proctex.color_diff_table.begin(), proctex.color_diff_table.end(),
                       new_data.begin(), [](const auto& entry) {
                           auto rgba = entry.ToVector() / 255.0f;
                           return Common::Vec4f{rgba.r(), rgba.g(), rgba.b(), rgba.a()};
                       });

        if (new_data != proctex_diff_lut_data || invalidate) {
            proctex_diff_lut_data = new_data;
            std::memcpy(buffer + bytes_used, new_data.data(),
                        new_data.size() * sizeof(Common::Vec4f));
            fs_uniform_block_data.data.proctex_diff_lut_offset =
                static_cast<int>((offset + bytes_used) / sizeof(Common::Vec4f));
            fs_uniform_block_data.dirty = true;
            bytes_used += new_data.size() * sizeof(Common::Vec4f);
        }
        fs_uniform_block_data.proctex_diff_lut_dirty = false;
    }

    texture_buffer.Commit(static_cast<u32>(bytes_used));
}

void RasterizerVulkan::UploadUniforms(bool accelerate_draw) {
    const bool sync_vs_pica = accelerate_draw;
    const bool sync_vs = vs_uniform_block_data.dirty;
    const bool sync_fs = fs_uniform_block_data.dirty;
    if (!sync_vs_pica && !sync_vs && !sync_fs) {
        return;
    }

    const u32 uniform_size =
        uniform_size_aligned_vs_pica + uniform_size_aligned_vs + uniform_size_aligned_fs;
    auto [uniforms, offset, invalidate] =
        uniform_buffer.Map(uniform_size, uniform_buffer_alignment);

    u32 used_bytes = 0;

    if (sync_vs || invalidate) {
        std::memcpy(uniforms + used_bytes, &vs_uniform_block_data.data,
                    sizeof(vs_uniform_block_data.data));

        pipeline_cache.UpdateRange(1, offset + used_bytes);
        vs_uniform_block_data.dirty = false;
        used_bytes += uniform_size_aligned_vs;
    }

    if (sync_fs || invalidate) {
        std::memcpy(uniforms + used_bytes, &fs_uniform_block_data.data,
                    sizeof(fs_uniform_block_data.data));

        pipeline_cache.UpdateRange(2, offset + used_bytes);
        fs_uniform_block_data.dirty = false;
        used_bytes += uniform_size_aligned_fs;
    }

    if (sync_vs_pica) {
        VSPicaUniformData vs_uniforms;
        vs_uniforms.uniforms.SetFromRegs(regs.vs, pica.vs_setup);
        std::memcpy(uniforms + used_bytes, &vs_uniforms, sizeof(vs_uniforms));

        pipeline_cache.UpdateRange(0, offset + used_bytes);
        used_bytes += uniform_size_aligned_vs_pica;
    }

    uniform_buffer.Commit(used_bytes);
}

} // namespace Vulkan
