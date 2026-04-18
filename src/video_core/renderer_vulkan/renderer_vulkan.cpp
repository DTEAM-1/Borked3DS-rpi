// Copyright 2023 Citra Emulator Project
// Copyright 2024 Borked3DS Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include "common/assert.h"
#include <algorithm>
#include <vector>
#include "common/color.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "common/logging/log.h"
#include "common/memory_detect.h"
#include "common/profiling.h"
#include "common/settings.h"
#include "core/core.h"
#include "core/frontend/emu_window.h"
#include "video_core/gpu.h"
#include "video_core/pica/pica_core.h"
#include "video_core/renderer_vulkan/renderer_vulkan.h"
#include "video_core/renderer_vulkan/vk_memory_util.h"
#include "video_core/renderer_vulkan/vk_shader_util.h"

#include "video_core/host_shaders/vulkan_present_anaglyph_dubois_frag.h"
#include "video_core/host_shaders/vulkan_present_anaglyph_rendepth_frag.h"
#include "video_core/host_shaders/vulkan_present_frag.h"
#include "video_core/host_shaders/vulkan_present_interlaced_frag.h"
#include "video_core/host_shaders/vulkan_present_vert.h"

#include <vk_mem_alloc.h>

namespace Vulkan {

namespace {

[[nodiscard]] bool IsPresentTraceEnabled() {
    const char* value = std::getenv("BORKED3DS_V3DV_TRACE_PRESENT");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

[[nodiscard]] bool IsStrictCompatEnabled() {
    const char* value = std::getenv("BORKED3DS_V3DV_STRICT_COMPAT");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

[[nodiscard]] bool IsRenderTargetTraceEnabled() {
    const char* value = std::getenv("BORKED3DS_V3DV_TRACE_RT");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

[[nodiscard]] u32 GetRenderTargetTraceFrameBudget() {
    const char* value = std::getenv("BORKED3DS_V3DV_TRACE_RT_FRAMES");
    if (value == nullptr || value[0] == '\0') {
        return 3;
    }
    const long parsed = std::strtol(value, nullptr, 10);
    return parsed > 0 ? static_cast<u32>(parsed) : 3u;
}


struct CpuScreenDump {
    std::vector<u8> rgba;
    u32 width = 0;
    u32 height = 0;
    bool valid = false;
    bool visible = false;
    bool alpha_only = false;
};

static std::array<CpuScreenDump, 3> g_strict_compat_cpu_screens{};

struct RenderTargetTraceStats;
[[nodiscard]] bool StatsHaveVisibleRGB(const RenderTargetTraceStats& stats);
[[nodiscard]] bool StatsAreAlphaOnly(const RenderTargetTraceStats& stats);

void DecodeFramebufferRGB8AsBGR(const u8* framebuffer_data, u32 width, u32 height, u32 pixel_stride,
                                std::vector<u8>& rgba) {
    const u32 copy_width = std::min(width, pixel_stride);
    rgba.assign(static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 4, 0);
    for (u32 y = 0; y < height; ++y) {
        const u8* src_row = framebuffer_data + (static_cast<std::size_t>(y) * pixel_stride * 3);
        for (u32 x = 0; x < copy_width; ++x) {
            const std::size_t src = static_cast<std::size_t>(x) * 3;
            const std::size_t dst = (static_cast<std::size_t>(y) * width + x) * 4;
            rgba[dst + 0] = src_row[src + 2];
            rgba[dst + 1] = src_row[src + 1];
            rgba[dst + 2] = src_row[src + 0];
            rgba[dst + 3] = 255;
        }
    }
}


[[nodiscard]] bool DecodeFramebufferToRGBA8(const u8* framebuffer_data, u32 width, u32 height,
                                            u32 pixel_stride, Pica::PixelFormat format,
                                            std::vector<u8>& rgba) {
    if (framebuffer_data == nullptr || width == 0 || height == 0 || pixel_stride == 0) {
        return false;
    }

    const u32 copy_width = std::min(width, pixel_stride);
    rgba.assign(static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 4, 0);

    auto write_pixel = [&](u32 x, u32 y, u8 r, u8 g, u8 b, u8 a) {
        const std::size_t offset =
            (static_cast<std::size_t>(y) * static_cast<std::size_t>(width) +
             static_cast<std::size_t>(x)) *
            4;
        rgba[offset + 0] = r;
        rgba[offset + 1] = g;
        rgba[offset + 2] = b;
        rgba[offset + 3] = a;
    };

    switch (format) {
    case Pica::PixelFormat::RGBA8:
        for (u32 y = 0; y < height; ++y) {
            const u8* src_row = framebuffer_data + (static_cast<std::size_t>(y) * pixel_stride * 4);
            for (u32 x = 0; x < copy_width; ++x) {
                const auto color = Common::Color::DecodeRGBA8(src_row + static_cast<std::size_t>(x) * 4);
                write_pixel(x, y, color.r(), color.g(), color.b(), color.a());
            }
        }
        return true;

    case Pica::PixelFormat::RGB8:
        for (u32 y = 0; y < height; ++y) {
            const u8* src_row = framebuffer_data + (static_cast<std::size_t>(y) * pixel_stride * 3);
            for (u32 x = 0; x < copy_width; ++x) {
                const auto color = Common::Color::DecodeRGB8(src_row + static_cast<std::size_t>(x) * 3);
                write_pixel(x, y, color.r(), color.g(), color.b(), 255);
            }
        }
        return true;

    case Pica::PixelFormat::RGB565:
        for (u32 y = 0; y < height; ++y) {
            const u8* src_row = framebuffer_data + (static_cast<std::size_t>(y) * pixel_stride * 2);
            for (u32 x = 0; x < copy_width; ++x) {
                const auto color = Common::Color::DecodeRGB565(src_row + static_cast<std::size_t>(x) * 2);
                write_pixel(x, y, color.r(), color.g(), color.b(), 255);
            }
        }
        return true;

    case Pica::PixelFormat::RGB5A1:
        for (u32 y = 0; y < height; ++y) {
            const u8* src_row = framebuffer_data + (static_cast<std::size_t>(y) * pixel_stride * 2);
            for (u32 x = 0; x < copy_width; ++x) {
                const auto color = Common::Color::DecodeRGB5A1(src_row + static_cast<std::size_t>(x) * 2);
                write_pixel(x, y, color.r(), color.g(), color.b(), color.a());
            }
        }
        return true;

    case Pica::PixelFormat::RGBA4:
        for (u32 y = 0; y < height; ++y) {
            const u8* src_row = framebuffer_data + (static_cast<std::size_t>(y) * pixel_stride * 2);
            for (u32 x = 0; x < copy_width; ++x) {
                const auto color = Common::Color::DecodeRGBA4(src_row + static_cast<std::size_t>(x) * 2);
                write_pixel(x, y, color.r(), color.g(), color.b(), color.a());
            }
        }
        return true;

    default:
        return false;
    }
}


[[nodiscard]] bool EnsureRGBA8PresentTexture(const Instance& instance, PresentWindow& main_window,
                                             TextureInfo& texture, u32 width, u32 height) {
    if (width == 0 || height == 0) {
        return false;
    }

    vk::Device device = instance.GetDevice();
    if (texture.image_view && texture.width == width && texture.height == height &&
        texture.format == Pica::PixelFormat::RGBA8) {
        return true;
    }

    if (texture.image_view) {
        main_window.WaitPresent();
        device.destroyImageView(texture.image_view);
        texture.image_view = nullptr;
    }
    if (texture.image) {
        main_window.WaitPresent();
        vmaDestroyImage(instance.GetAllocator(), texture.image, texture.allocation);
        texture.image = nullptr;
        texture.allocation = nullptr;
    }

    const vk::ImageCreateInfo image_info = {
        .imageType = vk::ImageType::e2D,
        .format = vk::Format::eR8G8B8A8Unorm,
        .extent = {width, height, 1},
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = vk::SampleCountFlagBits::e1,
        .tiling = vk::ImageTiling::eOptimal,
        .usage = vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst |
        vk::ImageUsageFlagBits::eTransferSrc,
        .sharingMode = vk::SharingMode::eExclusive,
        .initialLayout = vk::ImageLayout::eUndefined,
    };

    const VmaAllocationCreateInfo alloc_info = {
        .flags = VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT,
        .usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
        .requiredFlags = 0,
        .preferredFlags = 0,
        .pool = VK_NULL_HANDLE,
        .pUserData = nullptr,
    };

    VkImage unsafe_image{};
    VkImageCreateInfo unsafe_image_info = static_cast<VkImageCreateInfo>(image_info);
    const VkResult result = vmaCreateImage(instance.GetAllocator(), &unsafe_image_info, &alloc_info,
                                           &unsafe_image, &texture.allocation, nullptr);
    if (result != VK_SUCCESS) [[unlikely]] {
        LOG_CRITICAL(Render_Vulkan, "Failed allocating present texture with error {}", result);
        return false;
    }
    texture.image = vk::Image{unsafe_image};

    const vk::ImageViewCreateInfo view_info = {
        .image = texture.image,
        .viewType = vk::ImageViewType::e2D,
        .format = vk::Format::eR8G8B8A8Unorm,
        .subresourceRange{
            .aspectMask = vk::ImageAspectFlagBits::eColor,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1,
        },
    };

    texture.image_view = device.createImageView(view_info);
    texture.width = width;
    texture.height = height;
    texture.format = Pica::PixelFormat::RGBA8;
    return true;
}

void BlitCpuScreenToCanvas(const CpuScreenDump& screen, std::vector<u8>& canvas, u32 canvas_width,
                           u32 canvas_height, const Common::Rectangle<u32>& rect,
                           Layout::DisplayOrientation orientation) {
    if (!screen.valid || screen.rgba.empty() || canvas.empty()) {
        return;
    }

    const u32 dst_w = rect.GetWidth();
    const u32 dst_h = rect.GetHeight();
    if (dst_w == 0 || dst_h == 0) {
        return;
    }

    const auto sample = [&](u32 sx, u32 sy, std::size_t channel) -> u8 {
        sx = std::min(sx, screen.width - 1);
        sy = std::min(sy, screen.height - 1);
        const std::size_t index =
            (static_cast<std::size_t>(sy) * static_cast<std::size_t>(screen.width) +
             static_cast<std::size_t>(sx)) * 4 + channel;
        return screen.rgba[index];
    };

    for (u32 dy = 0; dy < dst_h; ++dy) {
        const u32 py = rect.top + dy;
        if (py >= canvas_height) {
            continue;
        }
        for (u32 dx = 0; dx < dst_w; ++dx) {
            const u32 px = rect.left + dx;
            if (px >= canvas_width) {
                continue;
            }

            u32 sx = 0;
            u32 sy = 0;
            switch (orientation) {
            case Layout::DisplayOrientation::Landscape:
                sx = static_cast<u32>((static_cast<u64>(dx) * screen.width) / std::max<u32>(1, dst_w));
                sy = static_cast<u32>((static_cast<u64>(dy) * screen.height) / std::max<u32>(1, dst_h));
                break;
            case Layout::DisplayOrientation::Portrait:
                sx = static_cast<u32>((static_cast<u64>(dy) * screen.width) / std::max<u32>(1, dst_h));
                sy = screen.height - 1 -
                     static_cast<u32>((static_cast<u64>(dx) * screen.height) /
                                      std::max<u32>(1, dst_w));
                break;
            case Layout::DisplayOrientation::LandscapeFlipped:
                sx = screen.width - 1 -
                     static_cast<u32>((static_cast<u64>(dx) * screen.width) / std::max<u32>(1, dst_w));
                sy = screen.height - 1 -
                     static_cast<u32>((static_cast<u64>(dy) * screen.height) / std::max<u32>(1, dst_h));
                break;
            case Layout::DisplayOrientation::PortraitFlipped:
                sx = screen.width - 1 -
                     static_cast<u32>((static_cast<u64>(dy) * screen.width) / std::max<u32>(1, dst_h));
                sy = static_cast<u32>((static_cast<u64>(dx) * screen.height) / std::max<u32>(1, dst_w));
                break;
            default:
                break;
            }

            const std::size_t dst_index =
                (static_cast<std::size_t>(py) * static_cast<std::size_t>(canvas_width) +
                 static_cast<std::size_t>(px)) * 4;
            canvas[dst_index + 0] = sample(sx, sy, 0);
            canvas[dst_index + 1] = sample(sx, sy, 1);
            canvas[dst_index + 2] = sample(sx, sy, 2);
            canvas[dst_index + 3] = sample(sx, sy, 3);
        }
    }
}

[[nodiscard]] bool ComposeStrictCompatWindowCanvas(const Layout::FramebufferLayout& layout,
                                                   std::vector<u8>& canvas) {
    if (Settings::values.render_3d.GetValue() != Settings::StereoRenderOption::Off) {
        return false;
    }

    canvas.assign(static_cast<std::size_t>(layout.width) * static_cast<std::size_t>(layout.height) * 4, 0);

    const auto orientation = layout.is_rotated ? Layout::DisplayOrientation::Landscape
                                               : Layout::DisplayOrientation::Portrait;

    const int mono_eye = static_cast<int>(Settings::values.mono_render_option.GetValue());
    const u32 top_screen_id = mono_eye >= 0 && mono_eye < 2 ? static_cast<u32>(mono_eye) : 0;
    const u32 bottom_screen_id = 2;

    const auto draw_top = [&](const Common::Rectangle<u32>& rect) {
        BlitCpuScreenToCanvas(g_strict_compat_cpu_screens[top_screen_id], canvas, layout.width,
                              layout.height, rect, orientation);
    };
    const auto draw_bottom = [&](const Common::Rectangle<u32>& rect) {
        BlitCpuScreenToCanvas(g_strict_compat_cpu_screens[bottom_screen_id], canvas, layout.width,
                              layout.height, rect, orientation);
    };

    if (!Settings::values.swap_screen.GetValue()) {
        if (layout.top_screen_enabled) {
            draw_top(layout.top_screen);
        }
        if (layout.bottom_screen_enabled) {
            draw_bottom(layout.bottom_screen);
        }
    } else {
        if (layout.bottom_screen_enabled) {
            draw_bottom(layout.bottom_screen);
        }
        if (layout.top_screen_enabled) {
            draw_top(layout.top_screen);
        }
    }

    if (layout.additional_screen_enabled) {
        if (!Settings::values.swap_screen.GetValue()) {
            draw_top(layout.additional_screen);
        } else {
            draw_bottom(layout.additional_screen);
        }
    }

    return true;
}

[[nodiscard]] bool UploadRGBA8ToImage(const Instance& instance, Scheduler& scheduler,
                                      vk::Image dst_image, u32 width, u32 height,
                                      const std::vector<u8>& rgba) {
    if (!dst_image || width == 0 || height == 0 || rgba.empty()) {
        return false;
    }

    const vk::BufferCreateInfo staging_buffer_info = {
        .size = static_cast<vk::DeviceSize>(rgba.size()),
        .usage = vk::BufferUsageFlagBits::eTransferSrc,
    };

    const VmaAllocationCreateInfo alloc_create_info = {
        .flags = VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT |
                 VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
        .usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
        .requiredFlags = 0,
        .preferredFlags = 0,
        .pool = VK_NULL_HANDLE,
        .pUserData = nullptr,
    };

    VkBuffer unsafe_buffer{};
    VmaAllocation allocation{};
    VmaAllocationInfo alloc_info{};
    VkBufferCreateInfo unsafe_buffer_info = static_cast<VkBufferCreateInfo>(staging_buffer_info);

    const VkResult result = vmaCreateBuffer(instance.GetAllocator(), &unsafe_buffer_info,
                                            &alloc_create_info, &unsafe_buffer, &allocation,
                                            &alloc_info);
    if (result != VK_SUCCESS) {
        return false;
    }

    std::memcpy(alloc_info.pMappedData, rgba.data(), rgba.size());
    const vk::Buffer staging_buffer{unsafe_buffer};

    scheduler.Record([dst_image, staging_buffer, width, height](vk::CommandBuffer cmdbuf) {
        const vk::ImageSubresourceRange range = {
            .aspectMask = vk::ImageAspectFlagBits::eColor,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1,
        };

        const vk::ImageMemoryBarrier pre_barrier = {
            .srcAccessMask = vk::AccessFlagBits::eNone,
            .dstAccessMask = vk::AccessFlagBits::eTransferWrite,
            .oldLayout = vk::ImageLayout::eUndefined,
            .newLayout = vk::ImageLayout::eTransferDstOptimal,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = dst_image,
            .subresourceRange = range,
        };

        const vk::BufferImageCopy image_copy = {
            .bufferOffset = 0,
            .bufferRowLength = 0,
            .bufferImageHeight = 0,
            .imageSubresource = {
                .aspectMask = vk::ImageAspectFlagBits::eColor,
                .mipLevel = 0,
                .baseArrayLayer = 0,
                .layerCount = 1,
            },
            .imageOffset = {0, 0, 0},
            .imageExtent = {width, height, 1},
        };

        const vk::ImageMemoryBarrier post_barrier = {
            .srcAccessMask = vk::AccessFlagBits::eTransferWrite,
            .dstAccessMask = vk::AccessFlagBits::eShaderRead,
            .oldLayout = vk::ImageLayout::eTransferDstOptimal,
            .newLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = dst_image,
            .subresourceRange = range,
        };

        cmdbuf.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe,
                               vk::PipelineStageFlagBits::eTransfer,
                               vk::DependencyFlagBits::eByRegion, {}, {}, pre_barrier);
        cmdbuf.copyBufferToImage(staging_buffer, dst_image, vk::ImageLayout::eTransferDstOptimal,
                                 image_copy);
        cmdbuf.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                               vk::PipelineStageFlagBits::eFragmentShader,
                               vk::DependencyFlagBits::eByRegion, {}, {}, post_barrier);
    });

    scheduler.Finish();
    vmaDestroyBuffer(instance.GetAllocator(), staging_buffer, allocation);
    return true;
}

[[nodiscard]] bool UploadRGBA8DirectlyToFrameImage(const Instance& instance, Scheduler& scheduler,
                                                   vk::Image dst_image, u32 width, u32 height,
                                                   const std::vector<u8>& rgba) {
    if (!dst_image || width == 0 || height == 0 ||
        rgba.size() < static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 4) {
        return false;
    }

    const vk::DeviceSize buffer_size = static_cast<vk::DeviceSize>(rgba.size());
    const vk::BufferCreateInfo staging_buffer_info = {
        .size = buffer_size,
        .usage = vk::BufferUsageFlagBits::eTransferSrc,
        .sharingMode = vk::SharingMode::eExclusive,
    };

    const VmaAllocationCreateInfo alloc_create_info = {
        .flags = VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT |
                 VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
        .usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
        .requiredFlags = 0,
        .preferredFlags = 0,
        .pool = VK_NULL_HANDLE,
        .pUserData = nullptr,
    };

    VkBuffer unsafe_buffer{};
    VmaAllocation allocation{};
    VmaAllocationInfo alloc_info{};
    VkBufferCreateInfo unsafe_buffer_info = static_cast<VkBufferCreateInfo>(staging_buffer_info);

    const VkResult result = vmaCreateBuffer(instance.GetAllocator(), &unsafe_buffer_info,
                                            &alloc_create_info, &unsafe_buffer, &allocation,
                                            &alloc_info);
    if (result != VK_SUCCESS) {
        return false;
    }

    std::memcpy(alloc_info.pMappedData, rgba.data(), rgba.size());
    const vk::Buffer staging_buffer{unsafe_buffer};

    scheduler.Record([dst_image, staging_buffer, width, height](vk::CommandBuffer cmdbuf) {
        const vk::ImageSubresourceRange range = {
            .aspectMask = vk::ImageAspectFlagBits::eColor,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1,
        };

        const vk::ImageMemoryBarrier pre_barrier = {
            .srcAccessMask = vk::AccessFlagBits::eTransferRead | vk::AccessFlagBits::eMemoryRead,
            .dstAccessMask = vk::AccessFlagBits::eTransferWrite,
            .oldLayout = vk::ImageLayout::eTransferSrcOptimal,
            .newLayout = vk::ImageLayout::eTransferDstOptimal,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = dst_image,
            .subresourceRange = range,
        };

        const vk::BufferImageCopy image_copy = {
            .bufferOffset = 0,
            .bufferRowLength = 0,
            .bufferImageHeight = 0,
            .imageSubresource = {
                .aspectMask = vk::ImageAspectFlagBits::eColor,
                .mipLevel = 0,
                .baseArrayLayer = 0,
                .layerCount = 1,
            },
            .imageOffset = {0, 0, 0},
            .imageExtent = {width, height, 1},
        };

        const vk::ImageMemoryBarrier post_barrier = {
            .srcAccessMask = vk::AccessFlagBits::eTransferWrite,
            .dstAccessMask = vk::AccessFlagBits::eTransferRead | vk::AccessFlagBits::eMemoryRead,
            .oldLayout = vk::ImageLayout::eTransferDstOptimal,
            .newLayout = vk::ImageLayout::eTransferSrcOptimal,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = dst_image,
            .subresourceRange = range,
        };

        cmdbuf.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
                               vk::PipelineStageFlagBits::eTransfer,
                               vk::DependencyFlagBits::eByRegion, {}, {}, pre_barrier);
        cmdbuf.copyBufferToImage(staging_buffer, dst_image, vk::ImageLayout::eTransferDstOptimal,
                                 image_copy);
        cmdbuf.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                               vk::PipelineStageFlagBits::eAllCommands,
                               vk::DependencyFlagBits::eByRegion, {}, {}, post_barrier);
    });

    scheduler.Finish();
    vmaDestroyBuffer(instance.GetAllocator(), staging_buffer, allocation);
    return true;
}

void DirectCopyPresentTextureToFrame(Scheduler& scheduler, vk::Image src_image, vk::Image dst_image,
                                    u32 src_width, u32 src_height, u32 dst_width, u32 dst_height) {
    if (!src_image || !dst_image || src_width == 0 || src_height == 0 || dst_width == 0 || dst_height == 0) {
        return;
    }

    scheduler.Record([src_image, dst_image, src_width, src_height, dst_width, dst_height](vk::CommandBuffer cmdbuf) {
        const vk::ImageSubresourceRange range = {
            .aspectMask = vk::ImageAspectFlagBits::eColor,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1,
        };

        const vk::ImageMemoryBarrier src_pre_barrier = {
            .srcAccessMask = vk::AccessFlagBits::eShaderRead,
            .dstAccessMask = vk::AccessFlagBits::eTransferRead,
            .oldLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
            .newLayout = vk::ImageLayout::eTransferSrcOptimal,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = src_image,
            .subresourceRange = range,
        };

        const vk::ImageMemoryBarrier dst_pre_barrier = {
            .srcAccessMask = vk::AccessFlagBits::eNone,
            .dstAccessMask = vk::AccessFlagBits::eTransferWrite,
            .oldLayout = vk::ImageLayout::eUndefined,
            .newLayout = vk::ImageLayout::eTransferDstOptimal,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = dst_image,
            .subresourceRange = range,
        };

        const vk::ImageCopy image_copy = {
            .srcSubresource = {
                .aspectMask = vk::ImageAspectFlagBits::eColor,
                .mipLevel = 0,
                .baseArrayLayer = 0,
                .layerCount = 1,
            },
            .srcOffset = {0, 0, 0},
            .dstSubresource = {
                .aspectMask = vk::ImageAspectFlagBits::eColor,
                .mipLevel = 0,
                .baseArrayLayer = 0,
                .layerCount = 1,
            },
            .dstOffset = {0, 0, 0},
            .extent = {
                std::min(src_width, dst_width),
                std::min(src_height, dst_height),
                1,
            },
        };

        const vk::ImageMemoryBarrier src_post_barrier = {
            .srcAccessMask = vk::AccessFlagBits::eTransferRead,
            .dstAccessMask = vk::AccessFlagBits::eShaderRead,
            .oldLayout = vk::ImageLayout::eTransferSrcOptimal,
            .newLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = src_image,
            .subresourceRange = range,
        };

        const vk::ImageMemoryBarrier dst_post_barrier = {
            .srcAccessMask = vk::AccessFlagBits::eTransferWrite,
            .dstAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite,
            .oldLayout = vk::ImageLayout::eTransferDstOptimal,
            .newLayout = vk::ImageLayout::eTransferSrcOptimal,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = dst_image,
            .subresourceRange = range,
        };

        cmdbuf.pipelineBarrier(vk::PipelineStageFlagBits::eFragmentShader,
                               vk::PipelineStageFlagBits::eTransfer,
                               vk::DependencyFlagBits::eByRegion, {}, {},
                               std::array{src_pre_barrier, dst_pre_barrier});
        cmdbuf.copyImage(src_image, vk::ImageLayout::eTransferSrcOptimal,
                         dst_image, vk::ImageLayout::eTransferDstOptimal,
                         image_copy);
        cmdbuf.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                               vk::PipelineStageFlagBits::eAllCommands,
                               vk::DependencyFlagBits::eByRegion, {}, {},
                               std::array{src_post_barrier, dst_post_barrier});
    });
}

struct RenderTargetTraceStats {
    u64 nonzero_pixels = 0;
    u64 alpha_nonzero_pixels = 0;
    u64 opaque_pixels = 0;
    u64 sum_r = 0;
    u64 sum_g = 0;
    u64 sum_b = 0;
    u64 sum_a = 0;
    u32 sample_count = 0;
    u32 width = 0;
    u32 height = 0;
};

[[nodiscard]] bool StatsHaveVisibleRGB(const RenderTargetTraceStats& stats) {
    return stats.sum_r != 0 || stats.sum_g != 0 || stats.sum_b != 0;
}

[[nodiscard]] bool StatsAreAlphaOnly(const RenderTargetTraceStats& stats) {
    return !StatsHaveVisibleRGB(stats) && stats.sum_a != 0 && stats.nonzero_pixels != 0;
}


[[nodiscard]] RenderTargetTraceStats AnalyzeRenderTargetRGBA8(const u8* rgba, u32 width, u32 height) {
    RenderTargetTraceStats stats{};
    stats.width = width;
    stats.height = height;
    if (rgba == nullptr || width == 0 || height == 0) {
        return stats;
    }

    const std::size_t pixel_count = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
    const std::size_t max_samples = std::min<std::size_t>(pixel_count, 4096);
    for (std::size_t i = 0; i < max_samples; ++i) {
        const std::size_t base = i * 4;
        const u8 r = rgba[base + 0];
        const u8 g = rgba[base + 1];
        const u8 b = rgba[base + 2];
        const u8 a = rgba[base + 3];
        stats.sample_count++;
        stats.sum_r += r;
        stats.sum_g += g;
        stats.sum_b += b;
        stats.sum_a += a;
        if (r != 0 || g != 0 || b != 0 || a != 0) {
            stats.nonzero_pixels++;
        }
        if (a != 0) {
            stats.alpha_nonzero_pixels++;
        }
        if (a == 255) {
            stats.opaque_pixels++;
        }
    }
    return stats;
}

[[nodiscard]] RenderTargetTraceStats AnalyzeRGBA8Vector(const std::vector<u8>& rgba, u32 width,
                                                      u32 height) {
    if (rgba.empty()) {
        return {};
    }
    return AnalyzeRenderTargetRGBA8(rgba.data(), width, height);
}

void InjectStrictCompatDebugOverlay(std::vector<u8>& rgba, u32 width, u32 height) {
    if (rgba.size() < static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 4 ||
        width == 0 || height == 0) {
        return;
    }

    const u32 marker_w = std::min<u32>(width, 64);
    const u32 marker_h = std::min<u32>(height, 64);
    for (u32 y = 0; y < marker_h; ++y) {
        for (u32 x = 0; x < marker_w; ++x) {
            const std::size_t base = (static_cast<std::size_t>(y) * width + x) * 4;
            const bool checker = ((x / 8) ^ (y / 8)) & 1;
            rgba[base + 0] = checker ? 255 : 0;
            rgba[base + 1] = checker ? 0 : 255;
            rgba[base + 2] = 255;
            rgba[base + 3] = 255;
        }
    }

    for (u32 x = 0; x < width; ++x) {
        const std::size_t top = static_cast<std::size_t>(x) * 4;
        const std::size_t bottom = (static_cast<std::size_t>(height - 1) * width + x) * 4;
        rgba[top + 0] = 255; rgba[top + 1] = 255; rgba[top + 2] = 255; rgba[top + 3] = 255;
        rgba[bottom + 0] = 255; rgba[bottom + 1] = 255; rgba[bottom + 2] = 255; rgba[bottom + 3] = 255;
    }
    for (u32 y = 0; y < height; ++y) {
        const std::size_t left = (static_cast<std::size_t>(y) * width) * 4;
        const std::size_t right = (static_cast<std::size_t>(y) * width + (width - 1)) * 4;
        rgba[left + 0] = 255; rgba[left + 1] = 255; rgba[left + 2] = 255; rgba[left + 3] = 255;
        rgba[right + 0] = 255; rgba[right + 1] = 255; rgba[right + 2] = 255; rgba[right + 3] = 255;
    }
}

void MaybeWriteRenderTargetPPM(const u8* rgba, u32 width, u32 height, u64 trace_index) {
    if (rgba == nullptr || width == 0 || height == 0) {
        return;
    }
    const char* dump_env = std::getenv("BORKED3DS_V3DV_TRACE_RT_WRITE");
    if (dump_env == nullptr || dump_env[0] == '\0' || dump_env[0] == '0') {
        return;
    }

    char path[256];
    std::snprintf(path, sizeof(path), "/tmp/borked3ds_rt_main_%04llu.ppm",
                  static_cast<unsigned long long>(trace_index));
    std::FILE* file = std::fopen(path, "wb");
    if (file == nullptr) {
        LOG_INFO(Render_Vulkan, "TRACE_RT dump_open_failed path='{}'", path);
        return;
    }

    std::fprintf(file, "P6\n%u %u\n255\n", width, height);
    for (std::size_t i = 0, pixels = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
         i < pixels; ++i) {
        const u8 rgb[3] = {rgba[i * 4 + 0], rgba[i * 4 + 1], rgba[i * 4 + 2]};
        std::fwrite(rgb, 1, sizeof(rgb), file);
    }
    std::fclose(file);
    LOG_INFO(Render_Vulkan, "TRACE_RT dump_written path='{}' width={} height={}", path, width, height);
}

} // namespace


struct ScreenRectVertex {
    ScreenRectVertex() = default;
    ScreenRectVertex(float x, float y, float u, float v)
        : position{Common::MakeVec(x, y)}, tex_coord{Common::MakeVec(u, v)} {}

    Common::Vec2f position;
    Common::Vec2f tex_coord;
};

constexpr u32 VERTEX_BUFFER_SIZE = sizeof(ScreenRectVertex) * 8192;

constexpr std::array<f32, 4 * 4> MakeOrthographicMatrix(u32 width, u32 height) {
    // clang-format off
    return { 2.f / width, 0.f,         0.f, -1.f,
            0.f,         2.f / height, 0.f, -1.f,
            0.f,         0.f,          1.f,  0.f,
            0.f,         0.f,          0.f,  1.f};
    // clang-format on
}

constexpr static std::array<vk::DescriptorSetLayoutBinding, 1> PRESENT_BINDINGS = {{
    {0, vk::DescriptorType::eCombinedImageSampler, 3, vk::ShaderStageFlagBits::eFragment},
}};

RendererVulkan::RendererVulkan(Core::System& system, Pica::PicaCore& pica_,
                               Frontend::EmuWindow& window, Frontend::EmuWindow* secondary_window)
    : RendererBase{system, window, secondary_window}, memory{system.Memory()}, pica{pica_},
      instance{window, Settings::values.physical_device.GetValue()}, scheduler{instance},
      renderpass_cache{instance, scheduler}, main_window{window, instance, scheduler},
      vertex_buffer{instance, scheduler, vk::BufferUsageFlagBits::eVertexBuffer,
                    VERTEX_BUFFER_SIZE},
      update_queue{instance},
      rasterizer{
          memory,   pica,      system.CustomTexManager(), *this,        render_window,
          instance, scheduler, renderpass_cache,          update_queue, main_window.ImageCount()},
      present_heap{instance, scheduler.GetMasterSemaphore(), PRESENT_BINDINGS, 32} {
    CompileShaders();
    BuildLayouts();
    BuildPipelines();
    ReloadPipeline();
    if (secondary_window) {
        second_window = std::make_unique<PresentWindow>(*secondary_window, instance, scheduler);
    }
}

RendererVulkan::~RendererVulkan() {
    vk::Device device = instance.GetDevice();
    scheduler.Finish();
    main_window.WaitPresent();
    device.waitIdle();

    device.destroyShaderModule(present_vertex_shader);
    for (u32 i = 0; i < PRESENT_PIPELINES; i++) {
        device.destroyPipeline(present_pipelines[i]);
        device.destroyShaderModule(present_shaders[i]);
    }

    for (auto& sampler : present_samplers) {
        device.destroySampler(sampler);
    }

    for (auto& info : screen_infos) {
        device.destroyImageView(info.texture.image_view);
        vmaDestroyImage(instance.GetAllocator(), info.texture.image, info.texture.allocation);
    }
}

void RendererVulkan::Sync() {
    rasterizer.SyncEntireState();
}

void RendererVulkan::PrepareRendertarget() {
    const auto& framebuffer_config = pica.regs.framebuffer_config;
    const auto& regs_lcd = pica.regs_lcd;

    if (IsPresentTraceEnabled()) {
        LOG_INFO(Render_Vulkan, "TRACE_PRESENT prepare_rendertarget begin");
    }

    for (u32 i = 0; i < 3; i++) {
        const u32 fb_id = i == 2 ? 1 : 0;
        const auto& framebuffer = framebuffer_config[fb_id];
        auto& texture = screen_infos[i].texture;

        const auto color_fill = fb_id == 0 ? regs_lcd.color_fill_top : regs_lcd.color_fill_bottom;
        if (color_fill.is_enabled) {
            if (texture.width != framebuffer.width || texture.height != framebuffer.height ||
                texture.format != framebuffer.color_format) {
                if (!ConfigureFramebufferTexture(texture, framebuffer)) {
                    continue;
                }
            }

            screen_infos[i].image_view = texture.image_view;
            screen_infos[i].texcoords = {0.f, 0.f, 1.f, 1.f};
            FillScreen(color_fill.AsVector(), texture);
            continue;
        }

        if (IsStrictCompatEnabled() || texture.width != framebuffer.width ||
            texture.height != framebuffer.height || texture.format != framebuffer.color_format) {
            if (!ConfigureFramebufferTexture(texture, framebuffer)) {
                continue;
            }
        }

        const bool loaded = LoadFBToScreenInfo(framebuffer, screen_infos[i], i == 1);
        if (!loaded) {
            screen_infos[i].image_view = texture.image_view;
            screen_infos[i].texcoords = {0.f, 0.f, 1.f, 1.f};
        }
        if (IsPresentTraceEnabled()) {
            LOG_INFO(Render_Vulkan,
                     "TRACE_PRESENT screen_info index={} loaded={} view_valid={} size={}x{}",
                     i, loaded, static_cast<bool>(screen_infos[i].image_view),
                     framebuffer.width.Value(), framebuffer.height.Value());
        }
    }
}

void RendererVulkan::PrepareDraw(Frame* frame, const Layout::FramebufferLayout& layout) {
    if (IsPresentTraceEnabled()) {
        for (u32 index = 0; index < screen_infos.size(); index++) {
            const vk::ImageView display_view = screen_infos[index].image_view;
            const vk::ImageView owned_view = screen_infos[index].texture.image_view;
            const bool use_display = static_cast<bool>(display_view);
            const std::string_view source = use_display ? "display_view" : "owned_fallback";
            LOG_INFO(Render_Vulkan,
                     "TRACE_PRESENT prepare_draw select_view index={} source={} display_valid={} owned_valid={} strict_compat={}",
                     index, source, static_cast<bool>(display_view), static_cast<bool>(owned_view),
                     static_cast<u32>(IsStrictCompatEnabled()));
        }
    }

    const bool strict_compat = IsStrictCompatEnabled();
    const auto present_set = strict_compat ? vk::DescriptorSet{} : present_heap.Commit();
    if (!strict_compat) {
        const auto sampler = present_samplers[!Settings::values.filter_mode.GetValue()];
        for (u32 index = 0; index < screen_infos.size(); index++) {
            const vk::ImageView display_view = screen_infos[index].image_view;
            const vk::ImageView owned_view = screen_infos[index].texture.image_view;
            const vk::ImageView image_view = display_view ? display_view : owned_view;
            update_queue.AddImageSampler(present_set, 0, index, image_view, sampler);
        }
        update_queue.Flush();
    }

    renderpass_cache.EndRendering();
    scheduler.Record([this, layout, frame, present_set, renderpass = main_window.Renderpass(),
                      index = current_pipeline, strict_compat](vk::CommandBuffer cmdbuf) {
        const vk::Viewport viewport = {
            .x = 0.0f,
            .y = 0.0f,
            .width = static_cast<float>(layout.width),
            .height = static_cast<float>(layout.height),
            .minDepth = 0.0f,
            .maxDepth = 1.0f,
        };

        const vk::Rect2D scissor = {
            .offset = {0, 0},
            .extent = {layout.width, layout.height},
        };

        cmdbuf.setViewport(0, viewport);
        cmdbuf.setScissor(0, scissor);

        const vk::ClearValue clear{.color = clear_color};
        const vk::PipelineLayout pipeline_layout{*present_pipeline_layout};
        const vk::RenderPassBeginInfo renderpass_begin_info = {
            .renderPass = renderpass,
            .framebuffer = frame->framebuffer,
            .renderArea =
                vk::Rect2D{
                    .offset = {0, 0},
                    .extent = {frame->width, frame->height},
                },
            .clearValueCount = 1,
            .pClearValues = &clear,
        };

        cmdbuf.beginRenderPass(renderpass_begin_info, vk::SubpassContents::eInline);
        cmdbuf.bindPipeline(vk::PipelineBindPoint::eGraphics, present_pipelines[index]);
        if (!strict_compat) {
            cmdbuf.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipeline_layout, 0,
                                      present_set, {});
        }
    });
}

void RendererVulkan::RenderToWindow(PresentWindow& window, const Layout::FramebufferLayout& layout,
                                    bool flipped) {
    if (IsStrictCompatEnabled()) {
        window.WaitPresent();
        scheduler.Finish();
    }

    Frame* frame = window.GetRenderFrame();

    if (layout.width != frame->width || layout.height != frame->height) {
        window.WaitPresent();
        scheduler.Finish();
        window.RecreateFrame(frame, layout.width, layout.height);
    }

    DrawScreens(frame, layout, flipped);
    scheduler.Flush(frame->render_ready);

    window.Present(frame);
}

bool RendererVulkan::LoadFBToScreenInfo(const Pica::FramebufferConfig& framebuffer,
                                        ScreenInfo& screen_info, bool right_eye) {
    if (framebuffer.address_right1 == 0 || framebuffer.address_right2 == 0) {
        right_eye = false;
    }

    const PAddr framebuffer_addr =
        framebuffer.active_fb == 0 ? (right_eye ? framebuffer.address_right1 : framebuffer.address_left1)
                                   : (right_eye ? framebuffer.address_right2 : framebuffer.address_left2);

    const u32 bpp = Pica::BytesPerPixel(framebuffer.color_format);
    const std::size_t pixel_stride = bpp != 0 ? (framebuffer.stride / bpp) : 0;
    const bool stride_divisible = bpp != 0 && (pixel_stride * bpp == framebuffer.stride);
    const bool stride_aligned4 = stride_divisible && ((pixel_stride % 4) == 0);

    if (IsPresentTraceEnabled()) {
        LOG_INFO(
            Render_Vulkan,
            "TRACE_PRESENT load_fb_to_screen right_eye={} active_fb={} addr=0x{:08X} width={} height={} stride={} format={} bpp={} pixel_stride={} stride_divisible={} stride_aligned4={}",
            right_eye, framebuffer.active_fb, framebuffer_addr, framebuffer.width.Value(),
            framebuffer.height.Value(), framebuffer.stride,
            static_cast<u32>(framebuffer.color_format.Value()), bpp, pixel_stride,
            static_cast<u32>(stride_divisible), static_cast<u32>(stride_aligned4));
    }

    LOG_TRACE(Render_Vulkan, "0x{:08x} bytes from 0x{:08x}({}x{}), fmt {:x}",
              framebuffer.stride * framebuffer.height, framebuffer_addr,
              framebuffer.width.Value(), framebuffer.height.Value(), framebuffer.format);

    if (bpp == 0) {
        if (IsPresentTraceEnabled()) {
            LOG_INFO(Render_Vulkan,
                     "TRACE_PRESENT load_fb_to_screen abort reason=zero_bpp addr=0x{:08X} format={}",
                     framebuffer_addr, static_cast<u32>(framebuffer.color_format.Value()));
        }
        return false;
    }

    if (!stride_divisible) {
        if (IsPresentTraceEnabled()) {
            LOG_INFO(Render_Vulkan,
                     "TRACE_PRESENT load_fb_to_screen abort reason=non_divisible_stride addr=0x{:08X} stride={} bpp={} pixel_stride={}",
                     framebuffer_addr, framebuffer.stride, bpp, pixel_stride);
        }
        return false;
    }

    if (!stride_aligned4 && IsPresentTraceEnabled()) {
        LOG_INFO(Render_Vulkan,
                 "TRACE_PRESENT load_fb_to_screen permissive_unaligned_stride addr=0x{:08X} pixel_stride={}",
                 framebuffer_addr, pixel_stride);
    }

    if (IsStrictCompatEnabled()) {
        const u8* framebuffer_data = memory.GetPhysicalPointer(framebuffer_addr);
        std::vector<u8> rgba;
        bool decoded = DecodeFramebufferToRGBA8(
            framebuffer_data, framebuffer.width.Value(), framebuffer.height.Value(),
            static_cast<u32>(pixel_stride), framebuffer.color_format, rgba);
        auto cpu_stats = AnalyzeRGBA8Vector(rgba, framebuffer.width.Value(), framebuffer.height.Value());

        if (decoded && framebuffer.color_format == Pica::PixelFormat::RGB8 &&
            !StatsHaveVisibleRGB(cpu_stats) && cpu_stats.sum_a != 0) {
            std::vector<u8> bgr_rgba;
            DecodeFramebufferRGB8AsBGR(framebuffer_data, framebuffer.width.Value(), framebuffer.height.Value(),
                                       static_cast<u32>(pixel_stride), bgr_rgba);
            const auto bgr_stats = AnalyzeRGBA8Vector(bgr_rgba, framebuffer.width.Value(), framebuffer.height.Value());
            if (IsPresentTraceEnabled()) {
                LOG_INFO(Render_Vulkan,
                         "TRACE_PRESENT cpu_screen_rgb8_bgr_retry addr=0x{:08X} width={} height={} nonzero={} alpha_nonzero={} opaque={} sum_rgba=({}, {}, {}, {})",
                         framebuffer_addr, framebuffer.width.Value(), framebuffer.height.Value(),
                         static_cast<unsigned long long>(bgr_stats.nonzero_pixels),
                         static_cast<unsigned long long>(bgr_stats.alpha_nonzero_pixels),
                         static_cast<unsigned long long>(bgr_stats.opaque_pixels),
                         static_cast<unsigned long long>(bgr_stats.sum_r),
                         static_cast<unsigned long long>(bgr_stats.sum_g),
                         static_cast<unsigned long long>(bgr_stats.sum_b),
                         static_cast<unsigned long long>(bgr_stats.sum_a));
            }
            if (StatsHaveVisibleRGB(bgr_stats)) {
                rgba.swap(bgr_rgba);
                cpu_stats = bgr_stats;
            }
        }

        const bool cpu_visible = StatsHaveVisibleRGB(cpu_stats);
        const bool uploaded = decoded &&
                              UploadRGBA8ToImage(instance, scheduler, screen_info.texture.image,
                                                 framebuffer.width.Value(), framebuffer.height.Value(),
                                                 rgba);

        if (IsPresentTraceEnabled()) {
            LOG_INFO(Render_Vulkan,
                     "TRACE_PRESENT cpu_screen_stats addr=0x{:08X} width={} height={} decoded={} uploaded={} nonzero={} alpha_nonzero={} opaque={} sum_rgba=({}, {}, {}, {})",
                     framebuffer_addr, framebuffer.width.Value(), framebuffer.height.Value(),
                     static_cast<u32>(decoded), static_cast<u32>(uploaded),
                     static_cast<unsigned long long>(cpu_stats.nonzero_pixels),
                     static_cast<unsigned long long>(cpu_stats.alpha_nonzero_pixels),
                     static_cast<unsigned long long>(cpu_stats.opaque_pixels),
                     static_cast<unsigned long long>(cpu_stats.sum_r),
                     static_cast<unsigned long long>(cpu_stats.sum_g),
                     static_cast<unsigned long long>(cpu_stats.sum_b),
                     static_cast<unsigned long long>(cpu_stats.sum_a));
        }

        const std::size_t screen_index = static_cast<std::size_t>(&screen_info - screen_infos.data());
        if (screen_index < g_strict_compat_cpu_screens.size()) {
            if (uploaded && cpu_visible) {
                g_strict_compat_cpu_screens[screen_index].rgba = rgba;
                g_strict_compat_cpu_screens[screen_index].width = framebuffer.width.Value();
                g_strict_compat_cpu_screens[screen_index].height = framebuffer.height.Value();
                g_strict_compat_cpu_screens[screen_index].valid = true;
                g_strict_compat_cpu_screens[screen_index].visible = true;
                g_strict_compat_cpu_screens[screen_index].alpha_only = false;
            } else {
                g_strict_compat_cpu_screens[screen_index] = {};
                g_strict_compat_cpu_screens[screen_index].alpha_only = StatsAreAlphaOnly(cpu_stats);
            }
        }

        bool accelerated = false;
        if (!cpu_visible) {
            accelerated = rasterizer.AccelerateDisplay(framebuffer, framebuffer_addr,
                                                       static_cast<u32>(pixel_stride), screen_info);
            if (IsPresentTraceEnabled()) {
                LOG_INFO(Render_Vulkan,
                         "TRACE_PRESENT cpu_screen_black_fallback_host addr=0x{:08X} uploaded={} cpu_visible={} host_valid={}",
                         framebuffer_addr, static_cast<u32>(uploaded), static_cast<u32>(cpu_visible),
                         static_cast<u32>(accelerated && static_cast<bool>(screen_info.image_view)));
            }
        }

        if (!accelerated) {
            screen_info.image_view = uploaded ? screen_info.texture.image_view : vk::ImageView{};
            screen_info.texcoords = {0.f, 0.f, 1.f, 1.f};
        }

        if (IsPresentTraceEnabled()) {
            LOG_INFO(Render_Vulkan,
                     "TRACE_PRESENT load_fb_to_screen result accelerated={} cpu_upload={} decoded={} addr=0x{:08X} pixel_stride={} width={} height={}",
                     static_cast<u32>(accelerated), static_cast<u32>(uploaded), static_cast<u32>(decoded), framebuffer_addr,
                     pixel_stride, framebuffer.width.Value(), framebuffer.height.Value());
        }
        return accelerated || uploaded;
    }

    const bool accelerated =
        rasterizer.AccelerateDisplay(framebuffer, framebuffer_addr, static_cast<u32>(pixel_stride),
                                     screen_info);

    if (IsPresentTraceEnabled()) {
        LOG_INFO(Render_Vulkan,
                 "TRACE_PRESENT load_fb_to_screen result accelerated={} addr=0x{:08X} pixel_stride={}",
                 static_cast<u32>(accelerated), framebuffer_addr, pixel_stride);
    }

    return accelerated;
}


void RendererVulkan::CompileShaders() {
    const vk::Device device = instance.GetDevice();
    const std::string_view preamble =
        instance.IsImageArrayDynamicIndexSupported() ? "#define ARRAY_DYNAMIC_INDEX" : "";
    present_vertex_shader =
        Compile(HostShaders::VULKAN_PRESENT_VERT, vk::ShaderStageFlagBits::eVertex, device);
    present_shaders[0] = Compile(HostShaders::VULKAN_PRESENT_FRAG,
                                 vk::ShaderStageFlagBits::eFragment, device, preamble);
    present_shaders[1] = Compile(HostShaders::VULKAN_PRESENT_ANAGLYPH_RENDEPTH_FRAG,
                                 vk::ShaderStageFlagBits::eFragment, device, preamble);
    present_shaders[2] = Compile(HostShaders::VULKAN_PRESENT_ANAGLYPH_DUBOIS_FRAG,
                                 vk::ShaderStageFlagBits::eFragment, device, preamble);
    present_shaders[3] = Compile(HostShaders::VULKAN_PRESENT_INTERLACED_FRAG,
                                 vk::ShaderStageFlagBits::eFragment, device, preamble);

    auto properties = instance.GetPhysicalDevice().getProperties();
    for (std::size_t i = 0; i < present_samplers.size(); i++) {
        const vk::Filter filter_mode = i == 0 ? vk::Filter::eLinear : vk::Filter::eNearest;
        const vk::SamplerCreateInfo sampler_info = {
            .magFilter = filter_mode,
            .minFilter = filter_mode,
            .mipmapMode = vk::SamplerMipmapMode::eLinear,
            .addressModeU = vk::SamplerAddressMode::eClampToEdge,
            .addressModeV = vk::SamplerAddressMode::eClampToEdge,
            .anisotropyEnable = instance.IsAnisotropicFilteringSupported(),
            .maxAnisotropy = properties.limits.maxSamplerAnisotropy,
            .compareEnable = false,
            .compareOp = vk::CompareOp::eAlways,
            .borderColor = vk::BorderColor::eIntOpaqueBlack,
            .unnormalizedCoordinates = false,
        };

        present_samplers[i] = device.createSampler(sampler_info);
    }
}

void RendererVulkan::BuildLayouts() {
    const vk::PushConstantRange push_range = {
        .stageFlags = vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
        .offset = 0,
        .size = sizeof(PresentUniformData),
    };

    const auto descriptor_set_layout = present_heap.Layout();
    const vk::PipelineLayoutCreateInfo layout_info = {
        .setLayoutCount = 1,
        .pSetLayouts = &descriptor_set_layout,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &push_range,
    };
    present_pipeline_layout = instance.GetDevice().createPipelineLayoutUnique(layout_info);
}

void RendererVulkan::BuildPipelines() {
    const vk::VertexInputBindingDescription binding = {
        .binding = 0,
        .stride = sizeof(ScreenRectVertex),
        .inputRate = vk::VertexInputRate::eVertex,
    };

    const std::array attributes = {
        vk::VertexInputAttributeDescription{
            .location = 0,
            .binding = 0,
            .format = vk::Format::eR32G32Sfloat,
            .offset = offsetof(ScreenRectVertex, position),
        },
        vk::VertexInputAttributeDescription{
            .location = 1,
            .binding = 0,
            .format = vk::Format::eR32G32Sfloat,
            .offset = offsetof(ScreenRectVertex, tex_coord),
        },
    };

    const vk::PipelineVertexInputStateCreateInfo vertex_input_info = {
        .vertexBindingDescriptionCount = 1,
        .pVertexBindingDescriptions = &binding,
        .vertexAttributeDescriptionCount = static_cast<u32>(attributes.size()),
        .pVertexAttributeDescriptions = attributes.data(),
    };

    const vk::PipelineInputAssemblyStateCreateInfo input_assembly = {
        .topology = vk::PrimitiveTopology::eTriangleStrip,
        .primitiveRestartEnable = false,
    };

    const vk::PipelineRasterizationStateCreateInfo raster_state = {
        .depthClampEnable = false,
        .rasterizerDiscardEnable = false,
        .cullMode = vk::CullModeFlagBits::eNone,
        .frontFace = vk::FrontFace::eClockwise,
        .depthBiasEnable = false,
        .lineWidth = 1.0f,
    };

    const vk::PipelineMultisampleStateCreateInfo multisampling = {
        .rasterizationSamples = vk::SampleCountFlagBits::e1,
        .sampleShadingEnable = false,
    };

    const vk::PipelineColorBlendAttachmentState colorblend_attachment = {
        .blendEnable = false,
        .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
                          vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA,
    };

    const vk::PipelineColorBlendStateCreateInfo color_blending = {
        .logicOpEnable = false,
        .attachmentCount = 1,
        .pAttachments = &colorblend_attachment,
        .blendConstants = std::array{1.0f, 1.0f, 1.0f, 1.0f},
    };

    const vk::Viewport placeholder_viewport = vk::Viewport{0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f};
    const vk::Rect2D placeholder_scissor = vk::Rect2D{{0, 0}, {1, 1}};
    const vk::PipelineViewportStateCreateInfo viewport_info = {
        .viewportCount = 1,
        .pViewports = &placeholder_viewport,
        .scissorCount = 1,
        .pScissors = &placeholder_scissor,
    };

    const std::array dynamic_states = {
        vk::DynamicState::eViewport,
        vk::DynamicState::eScissor,
    };

    const vk::PipelineDynamicStateCreateInfo dynamic_info = {
        .dynamicStateCount = static_cast<u32>(dynamic_states.size()),
        .pDynamicStates = dynamic_states.data(),
    };

    const vk::PipelineDepthStencilStateCreateInfo depth_info = {
        .depthTestEnable = false,
        .depthWriteEnable = false,
        .depthCompareOp = vk::CompareOp::eAlways,
        .depthBoundsTestEnable = false,
        .stencilTestEnable = false,
    };

    for (u32 i = 0; i < PRESENT_PIPELINES; i++) {
        const std::array shader_stages = {
            vk::PipelineShaderStageCreateInfo{
                .stage = vk::ShaderStageFlagBits::eVertex,
                .module = present_vertex_shader,
                .pName = "main",
            },
            vk::PipelineShaderStageCreateInfo{
                .stage = vk::ShaderStageFlagBits::eFragment,
                .module = present_shaders[i],
                .pName = "main",
            },
        };

        const vk::GraphicsPipelineCreateInfo pipeline_info = {
            .stageCount = static_cast<u32>(shader_stages.size()),
            .pStages = shader_stages.data(),
            .pVertexInputState = &vertex_input_info,
            .pInputAssemblyState = &input_assembly,
            .pViewportState = &viewport_info,
            .pRasterizationState = &raster_state,
            .pMultisampleState = &multisampling,
            .pDepthStencilState = &depth_info,
            .pColorBlendState = &color_blending,
            .pDynamicState = &dynamic_info,
            .layout = *present_pipeline_layout,
            .renderPass = main_window.Renderpass(),
        };

        const auto [result, pipeline] =
            instance.GetDevice().createGraphicsPipeline({}, pipeline_info);
        ASSERT_MSG(result == vk::Result::eSuccess, "Unable to build present pipelines");
        present_pipelines[i] = pipeline;
    }
}

bool RendererVulkan::ConfigureFramebufferTexture(TextureInfo& texture,
                                                 const Pica::FramebufferConfig& framebuffer) {
    vk::Device device = instance.GetDevice();
    if (texture.image_view) {
        main_window.WaitPresent();
        device.destroyImageView(texture.image_view);
        texture.image_view = nullptr;
    }
    if (texture.image) {
        main_window.WaitPresent();
        vmaDestroyImage(instance.GetAllocator(), texture.image, texture.allocation);
        texture.image = nullptr;
        texture.allocation = nullptr;
    }

    const VideoCore::PixelFormat pixel_format =
        VideoCore::PixelFormatFromGPUPixelFormat(framebuffer.color_format);
    const vk::Format format = IsStrictCompatEnabled() ? vk::Format::eR8G8B8A8Unorm
                                                       : instance.GetTraits(pixel_format).native;

    const vk::ImageCreateInfo image_info = {
        .imageType = vk::ImageType::e2D,
        .format = format,
        .extent = {framebuffer.width, framebuffer.height, 1},
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = vk::SampleCountFlagBits::e1,
        .tiling = vk::ImageTiling::eOptimal,
        .usage = vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst |
        vk::ImageUsageFlagBits::eTransferSrc,
        .sharingMode = vk::SharingMode::eExclusive,
        .initialLayout = vk::ImageLayout::eUndefined,
    };

    const VmaAllocationCreateInfo alloc_info = {
        .flags = VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT,
        .usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
        .requiredFlags = 0,
        .preferredFlags = 0,
        .pool = VK_NULL_HANDLE,
        .pUserData = nullptr,
    };

    VkImage unsafe_image{};
    VkImageCreateInfo unsafe_image_info = static_cast<VkImageCreateInfo>(image_info);
    VkResult result = vmaCreateImage(instance.GetAllocator(), &unsafe_image_info, &alloc_info,
                                     &unsafe_image, &texture.allocation, nullptr);
    if (result != VK_SUCCESS) [[unlikely]] {
        LOG_CRITICAL(Render_Vulkan, "Failed allocating texture with error {}", result);
        return false;
    }
    texture.image = vk::Image{unsafe_image};

    const vk::ImageViewCreateInfo view_info = {
        .image = texture.image,
        .viewType = vk::ImageViewType::e2D,
        .format = format,
        .subresourceRange{
            .aspectMask = vk::ImageAspectFlagBits::eColor,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1,
        },
    };

    texture.image_view = device.createImageView(view_info);
    texture.width = framebuffer.width;
    texture.height = framebuffer.height;
    texture.format = framebuffer.color_format;
    return true;
}


void RendererVulkan::FillScreen(Common::Vec3<u8> color, const TextureInfo& texture) {
    const vk::ClearColorValue clear_color = {
        .float32 =
            std::array{
                color.r() / 255.0f,
                color.g() / 255.0f,
                color.b() / 255.0f,
                1.0f,
            },
    };

    renderpass_cache.EndRendering();
    scheduler.Record([image = texture.image, clear_color](vk::CommandBuffer cmdbuf) {
        const vk::ImageSubresourceRange range = {
            .aspectMask = vk::ImageAspectFlagBits::eColor,
            .baseMipLevel = 0,
            .levelCount = VK_REMAINING_MIP_LEVELS,
            .baseArrayLayer = 0,
            .layerCount = VK_REMAINING_ARRAY_LAYERS,
        };

        const vk::ImageMemoryBarrier pre_barrier = {
            .srcAccessMask = vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eTransferRead,
            .dstAccessMask = vk::AccessFlagBits::eTransferWrite,
            .oldLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
            .newLayout = vk::ImageLayout::eTransferDstOptimal,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = image,
            .subresourceRange = range,
        };

        const vk::ImageMemoryBarrier post_barrier = {
            .srcAccessMask = vk::AccessFlagBits::eTransferWrite,
            .dstAccessMask = vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eTransferRead,
            .oldLayout = vk::ImageLayout::eTransferDstOptimal,
            .newLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = image,
            .subresourceRange = range,
        };

        cmdbuf.pipelineBarrier(vk::PipelineStageFlagBits::eFragmentShader,
                               vk::PipelineStageFlagBits::eTransfer,
                               vk::DependencyFlagBits::eByRegion, {}, {}, pre_barrier);

        cmdbuf.clearColorImage(image, vk::ImageLayout::eTransferDstOptimal, clear_color, range);

        cmdbuf.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                               vk::PipelineStageFlagBits::eFragmentShader,
                               vk::DependencyFlagBits::eByRegion, {}, {}, post_barrier);
    });
}

void RendererVulkan::ReloadPipeline() {
    const Settings::StereoRenderOption render_3d = Settings::values.render_3d.GetValue();
    switch (render_3d) {
    case Settings::StereoRenderOption::Anaglyph:
        current_pipeline = 1;
        break;
    case Settings::StereoRenderOption::Interlaced:
    case Settings::StereoRenderOption::ReverseInterlaced:
        current_pipeline = 2;
        draw_info.reverse_interlaced = render_3d == Settings::StereoRenderOption::ReverseInterlaced;
        break;
    default:
        current_pipeline = 0;
        break;
    }
}

void RendererVulkan::DrawSingleScreen(u32 screen_id, float x, float y, float w, float h,
                                      Layout::DisplayOrientation orientation) {
    const ScreenInfo& screen_info = screen_infos[screen_id];
    const auto& texcoords = screen_info.texcoords;

    std::array<ScreenRectVertex, 4> vertices;
    switch (orientation) {
    case Layout::DisplayOrientation::Landscape:
        vertices = {{
            ScreenRectVertex(x, y, texcoords.bottom, texcoords.left),
            ScreenRectVertex(x + w, y, texcoords.bottom, texcoords.right),
            ScreenRectVertex(x, y + h, texcoords.top, texcoords.left),
            ScreenRectVertex(x + w, y + h, texcoords.top, texcoords.right),
        }};
        break;
    case Layout::DisplayOrientation::Portrait:
        vertices = {{
            ScreenRectVertex(x, y, texcoords.bottom, texcoords.right),
            ScreenRectVertex(x + w, y, texcoords.top, texcoords.right),
            ScreenRectVertex(x, y + h, texcoords.bottom, texcoords.left),
            ScreenRectVertex(x + w, y + h, texcoords.top, texcoords.left),
        }};
        std::swap(h, w);
        break;
    case Layout::DisplayOrientation::LandscapeFlipped:
        vertices = {{
            ScreenRectVertex(x, y, texcoords.top, texcoords.right),
            ScreenRectVertex(x + w, y, texcoords.top, texcoords.left),
            ScreenRectVertex(x, y + h, texcoords.bottom, texcoords.right),
            ScreenRectVertex(x + w, y + h, texcoords.bottom, texcoords.left),
        }};
        break;
    case Layout::DisplayOrientation::PortraitFlipped:
        vertices = {{
            ScreenRectVertex(x, y, texcoords.top, texcoords.left),
            ScreenRectVertex(x + w, y, texcoords.bottom, texcoords.left),
            ScreenRectVertex(x, y + h, texcoords.top, texcoords.right),
            ScreenRectVertex(x + w, y + h, texcoords.bottom, texcoords.right),
        }};
        std::swap(h, w);
        break;
    default:
        LOG_ERROR(Render_Vulkan, "Unknown DisplayOrientation: {}", orientation);
        break;
    }

    const u64 size = sizeof(ScreenRectVertex) * vertices.size();
    auto [data, offset, invalidate] = vertex_buffer.Map(size, 16);
    std::memcpy(data, vertices.data(), size);
    vertex_buffer.Commit(size);

    const u32 scale_factor = GetResolutionScaleFactor();
    draw_info.i_resolution =
        Common::MakeVec(static_cast<f32>(screen_info.texture.width * scale_factor),
                        static_cast<f32>(screen_info.texture.height * scale_factor),
                        1.0f / static_cast<f32>(screen_info.texture.width * scale_factor),
                        1.0f / static_cast<f32>(screen_info.texture.height * scale_factor));
    draw_info.o_resolution = Common::MakeVec(h, w, 1.0f / h, 1.0f / w);
    draw_info.screen_id_l = screen_id;

    const bool strict_compat = IsStrictCompatEnabled();
    const auto sampler = present_samplers[!Settings::values.filter_mode.GetValue()];
    const auto present_set = strict_compat ? present_heap.Commit() : vk::DescriptorSet{};
    if (strict_compat) {
        const vk::ImageView image_view = screen_info.image_view ? screen_info.image_view
                                                                : screen_info.texture.image_view;
        for (u32 i = 0; i < 3; ++i) {
            update_queue.AddImageSampler(present_set, 0, i, image_view, sampler);
        }
        update_queue.Flush();
        if (IsPresentTraceEnabled()) {
            LOG_INFO(Render_Vulkan,
                     "TRACE_PRESENT compat_bind_single screen_id={} duplicated_view_valid={} strict_compat=1",
                     screen_id, static_cast<bool>(image_view));
        }
    }

    scheduler.Record([this, offset = offset, info = draw_info, present_set, strict_compat](vk::CommandBuffer cmdbuf) {
        const u32 first_vertex = static_cast<u32>(offset) / sizeof(ScreenRectVertex);
        cmdbuf.pushConstants(*present_pipeline_layout,
                             vk::ShaderStageFlagBits::eFragment | vk::ShaderStageFlagBits::eVertex,
                             0, sizeof(info), &info);

        if (strict_compat) {
            cmdbuf.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *present_pipeline_layout,
                                      0, present_set, {});
        }
        cmdbuf.bindVertexBuffers(0, vertex_buffer.Handle(), {0});
        cmdbuf.draw(4, 1, first_vertex, 0);
    });
}

void RendererVulkan::DrawSingleScreenStereo(u32 screen_id_l, u32 screen_id_r, float x, float y,
                                            float w, float h,
                                            Layout::DisplayOrientation orientation) {
    const ScreenInfo& screen_info_l = screen_infos[screen_id_l];
    const auto& texcoords = screen_info_l.texcoords;

    std::array<ScreenRectVertex, 4> vertices;
    switch (orientation) {
    case Layout::DisplayOrientation::Landscape:
        vertices = {{
            ScreenRectVertex(x, y, texcoords.bottom, texcoords.left),
            ScreenRectVertex(x + w, y, texcoords.bottom, texcoords.right),
            ScreenRectVertex(x, y + h, texcoords.top, texcoords.left),
            ScreenRectVertex(x + w, y + h, texcoords.top, texcoords.right),
        }};
        break;
    case Layout::DisplayOrientation::Portrait:
        vertices = {{
            ScreenRectVertex(x, y, texcoords.bottom, texcoords.right),
            ScreenRectVertex(x + w, y, texcoords.top, texcoords.right),
            ScreenRectVertex(x, y + h, texcoords.bottom, texcoords.left),
            ScreenRectVertex(x + w, y + h, texcoords.top, texcoords.left),
        }};
        std::swap(h, w);
        break;
    case Layout::DisplayOrientation::LandscapeFlipped:
        vertices = {{
            ScreenRectVertex(x, y, texcoords.top, texcoords.right),
            ScreenRectVertex(x + w, y, texcoords.top, texcoords.left),
            ScreenRectVertex(x, y + h, texcoords.bottom, texcoords.right),
            ScreenRectVertex(x + w, y + h, texcoords.bottom, texcoords.left),
        }};
        break;
    case Layout::DisplayOrientation::PortraitFlipped:
        vertices = {{
            ScreenRectVertex(x, y, texcoords.top, texcoords.left),
            ScreenRectVertex(x + w, y, texcoords.bottom, texcoords.left),
            ScreenRectVertex(x, y + h, texcoords.top, texcoords.right),
            ScreenRectVertex(x + w, y + h, texcoords.bottom, texcoords.right),
        }};
        std::swap(h, w);
        break;
    default:
        LOG_ERROR(Render_Vulkan, "Unknown DisplayOrientation: {}", orientation);
        break;
    }

    const u64 size = sizeof(ScreenRectVertex) * vertices.size();
    auto [data, offset, invalidate] = vertex_buffer.Map(size, 16);
    std::memcpy(data, vertices.data(), size);
    vertex_buffer.Commit(size);

    const u32 scale_factor = GetResolutionScaleFactor();
    draw_info.i_resolution =
        Common::MakeVec(static_cast<f32>(screen_info_l.texture.width * scale_factor),
                        static_cast<f32>(screen_info_l.texture.height * scale_factor),
                        1.0f / static_cast<f32>(screen_info_l.texture.width * scale_factor),
                        1.0f / static_cast<f32>(screen_info_l.texture.height * scale_factor));
    draw_info.o_resolution = Common::MakeVec(h, w, 1.0f / h, 1.0f / w);
    draw_info.screen_id_l = screen_id_l;
    draw_info.screen_id_r = screen_id_r;

    const bool strict_compat = IsStrictCompatEnabled();
    const auto sampler = present_samplers[!Settings::values.filter_mode.GetValue()];
    const auto present_set = strict_compat ? present_heap.Commit() : vk::DescriptorSet{};
    if (strict_compat) {
        const vk::ImageView left_view = screen_infos[screen_id_l].image_view
                                            ? screen_infos[screen_id_l].image_view
                                            : screen_infos[screen_id_l].texture.image_view;
        const vk::ImageView right_view = screen_infos[screen_id_r].image_view
                                             ? screen_infos[screen_id_r].image_view
                                             : screen_infos[screen_id_r].texture.image_view;
        for (u32 i = 0; i < 3; ++i) {
            const vk::ImageView view = (i == screen_id_r) ? right_view : left_view;
            update_queue.AddImageSampler(present_set, 0, i, view, sampler);
        }
        update_queue.Flush();
        if (IsPresentTraceEnabled()) {
            LOG_INFO(Render_Vulkan,
                     "TRACE_PRESENT compat_bind_stereo screen_id_l={} screen_id_r={} left_valid={} right_valid={} strict_compat=1",
                     screen_id_l, screen_id_r, static_cast<bool>(left_view),
                     static_cast<bool>(right_view));
        }
    }

    scheduler.Record([this, offset = offset, info = draw_info, present_set, strict_compat](vk::CommandBuffer cmdbuf) {
        const u32 first_vertex = static_cast<u32>(offset) / sizeof(ScreenRectVertex);
        cmdbuf.pushConstants(*present_pipeline_layout,
                             vk::ShaderStageFlagBits::eFragment | vk::ShaderStageFlagBits::eVertex,
                             0, sizeof(info), &info);

        if (strict_compat) {
            cmdbuf.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *present_pipeline_layout,
                                      0, present_set, {});
        }
        cmdbuf.bindVertexBuffers(0, vertex_buffer.Handle(), {0});
        cmdbuf.draw(4, 1, first_vertex, 0);
    });
}

void RendererVulkan::DrawTopScreen(const Layout::FramebufferLayout& layout,
                                   const Common::Rectangle<u32>& top_screen) {
    if (!layout.top_screen_enabled) {
        return;
    }
    int leftside, rightside;
    leftside = Settings::values.swap_eyes_3d.GetValue() ? 1 : 0;
    rightside = Settings::values.swap_eyes_3d.GetValue() ? 0 : 1;
    const float top_screen_left = static_cast<float>(top_screen.left);
    const float top_screen_top = static_cast<float>(top_screen.top);
    const float top_screen_width = static_cast<float>(top_screen.GetWidth());
    const float top_screen_height = static_cast<float>(top_screen.GetHeight());

    const auto orientation = layout.is_rotated ? Layout::DisplayOrientation::Landscape
                                               : Layout::DisplayOrientation::Portrait;
    switch (Settings::values.render_3d.GetValue()) {
    case Settings::StereoRenderOption::Off: {
        const int eye = static_cast<int>(Settings::values.mono_render_option.GetValue());
        DrawSingleScreen(eye, top_screen_left, top_screen_top, top_screen_width, top_screen_height,
                         orientation);
        break;
    }
    case Settings::StereoRenderOption::SideBySide: {
        DrawSingleScreen(leftside, top_screen_left / 2, top_screen_top, top_screen_width / 2,
                         top_screen_height, orientation);
        draw_info.layer = 1;
        DrawSingleScreen(rightside, static_cast<float>((top_screen_left / 2) + (layout.width / 2)),
                         top_screen_top, top_screen_width / 2, top_screen_height, orientation);
        break;
    }
    case Settings::StereoRenderOption::SideBySideFull: {
        DrawSingleScreen(leftside, top_screen_left, top_screen_top, top_screen_width,
                         top_screen_height, orientation);
        draw_info.layer = 1;
        DrawSingleScreen(rightside, top_screen_left + layout.width / 2, top_screen_top,
                         top_screen_width, top_screen_height, orientation);
        break;
    }
    case Settings::StereoRenderOption::CardboardVR: {
        DrawSingleScreen(leftside, top_screen_left, top_screen_top, top_screen_width,
                         top_screen_height, orientation);
        draw_info.layer = 1;
        DrawSingleScreen(
            rightside,
            static_cast<float>(layout.cardboard.top_screen_right_eye + (layout.width / 2)),
            top_screen_top, top_screen_width, top_screen_height, orientation);
        break;
    }
    case Settings::StereoRenderOption::Anaglyph:
    case Settings::StereoRenderOption::Interlaced:
    case Settings::StereoRenderOption::ReverseInterlaced: {
        DrawSingleScreenStereo(leftside, rightside, top_screen_left, top_screen_top,
                               top_screen_width, top_screen_height, orientation);
        break;
    }
    }
}

void RendererVulkan::DrawBottomScreen(const Layout::FramebufferLayout& layout,
                                      const Common::Rectangle<u32>& bottom_screen) {
    if (!layout.bottom_screen_enabled) {
        return;
    }

    const float bottom_screen_left = static_cast<float>(bottom_screen.left);
    const float bottom_screen_top = static_cast<float>(bottom_screen.top);
    const float bottom_screen_width = static_cast<float>(bottom_screen.GetWidth());
    const float bottom_screen_height = static_cast<float>(bottom_screen.GetHeight());

    const auto orientation = layout.is_rotated ? Layout::DisplayOrientation::Landscape
                                               : Layout::DisplayOrientation::Portrait;

    bool separate_win = false;
#ifndef ANDROID
    separate_win =
        (Settings::values.layout_option.GetValue() == Settings::LayoutOption::SeparateWindows);
#endif

    switch (Settings::values.render_3d.GetValue()) {
    case Settings::StereoRenderOption::Off: {
        DrawSingleScreen(2, bottom_screen_left, bottom_screen_top, bottom_screen_width,
                         bottom_screen_height, orientation);
        break;
    }
    case Settings::StereoRenderOption::SideBySide: {
        if (separate_win) {
            DrawSingleScreen(2, bottom_screen_left, bottom_screen_top, bottom_screen_width,
                             bottom_screen_height, orientation);
        } else {

            DrawSingleScreen(2, bottom_screen_left / 2, bottom_screen_top, bottom_screen_width / 2,
                             bottom_screen_height, orientation);
            draw_info.layer = 1;
            DrawSingleScreen(2, static_cast<float>((bottom_screen_left / 2) + (layout.width / 2)),
                             bottom_screen_top, bottom_screen_width / 2, bottom_screen_height,
                             orientation);
        }
        break;
    }
    case Settings::StereoRenderOption::SideBySideFull: {
        DrawSingleScreen(2, bottom_screen_left, bottom_screen_top, bottom_screen_width,
                         bottom_screen_height, orientation);
        draw_info.layer = 1;
        DrawSingleScreen(2, bottom_screen_left + layout.width / 2, bottom_screen_top,
                         bottom_screen_width, bottom_screen_height, orientation);
        break;
    }
    case Settings::StereoRenderOption::CardboardVR: {
        DrawSingleScreen(2, bottom_screen_left, bottom_screen_top, bottom_screen_width,
                         bottom_screen_height, orientation);
        draw_info.layer = 1;
        DrawSingleScreen(
            2, static_cast<float>(layout.cardboard.bottom_screen_right_eye + (layout.width / 2)),
            bottom_screen_top, bottom_screen_width, bottom_screen_height, orientation);
        break;
    }
    case Settings::StereoRenderOption::Anaglyph:
    case Settings::StereoRenderOption::Interlaced:
    case Settings::StereoRenderOption::ReverseInterlaced: {
        if (separate_win) {
            DrawSingleScreen(2, bottom_screen_left, bottom_screen_top, bottom_screen_width,
                             bottom_screen_height, orientation);
        } else {
            DrawSingleScreenStereo(2, 2, bottom_screen_left, bottom_screen_top, bottom_screen_width,
                                   bottom_screen_height, orientation);
        }
        break;
    }
    }
}

void RendererVulkan::DrawScreens(Frame* frame, const Layout::FramebufferLayout& layout,
                                 bool flipped) {
    if (IsPresentTraceEnabled()) {
        static u64 trace_frame_counter = 0;
        ++trace_frame_counter;
        if (trace_frame_counter <= 8 || (trace_frame_counter % 120) == 0) {
            LOG_INFO(Render_Vulkan,
                     "TRACE_PRESENT draw_screens frame={} layout={}x{} top={} bottom={} additional={}",
                     trace_frame_counter, layout.width, layout.height,
                     layout.top_screen_enabled, layout.bottom_screen_enabled,
                     layout.additional_screen_enabled);
        }
    }
    if (settings.bg_color_update_requested.exchange(false)) {
        clear_color.float32[0] = Settings::values.bg_red.GetValue();
        clear_color.float32[1] = Settings::values.bg_green.GetValue();
        clear_color.float32[2] = Settings::values.bg_blue.GetValue();
    }
    if (settings.shader_update_requested.exchange(false)) {
        ReloadPipeline();
    }

    if (IsStrictCompatEnabled() &&
        Settings::values.render_3d.GetValue() == Settings::StereoRenderOption::Off) {
        const int mono_eye = static_cast<int>(Settings::values.mono_render_option.GetValue());
        const u32 top_screen_id = mono_eye >= 0 && mono_eye < 2 ? static_cast<u32>(mono_eye) : 0;
        const bool top_visible = !layout.top_screen_enabled || g_strict_compat_cpu_screens[top_screen_id].visible;
        const bool bottom_visible = !layout.bottom_screen_enabled || g_strict_compat_cpu_screens[2].visible;
        const bool additional_visible = !layout.additional_screen_enabled || g_strict_compat_cpu_screens[top_screen_id].visible;
        const bool can_cpu_compose = top_visible && bottom_visible && additional_visible;

        std::vector<u8> composed_rgba;
        if (can_cpu_compose && ComposeStrictCompatWindowCanvas(layout, composed_rgba)) {
            if (IsPresentTraceEnabled()) {
                LOG_INFO(Render_Vulkan,
                         "TRACE_PRESENT cpu_compose_window uploaded=1 width={} height={} top_valid={} bottom_valid={}",
                         layout.width, layout.height,
                         static_cast<u32>(g_strict_compat_cpu_screens[0].visible ||
                                          g_strict_compat_cpu_screens[1].visible),
                         static_cast<u32>(g_strict_compat_cpu_screens[2].visible));
            }

            const auto canvas_stats = AnalyzeRGBA8Vector(composed_rgba, layout.width, layout.height);
            if (IsPresentTraceEnabled()) {
                LOG_INFO(Render_Vulkan,
                         "TRACE_PRESENT cpu_canvas_stats width={} height={} nonzero={} alpha_nonzero={} opaque={} sum_rgba=({}, {}, {}, {})",
                         layout.width, layout.height,
                         static_cast<unsigned long long>(canvas_stats.nonzero_pixels),
                         static_cast<unsigned long long>(canvas_stats.alpha_nonzero_pixels),
                         static_cast<unsigned long long>(canvas_stats.opaque_pixels),
                         static_cast<unsigned long long>(canvas_stats.sum_r),
                         static_cast<unsigned long long>(canvas_stats.sum_g),
                         static_cast<unsigned long long>(canvas_stats.sum_b),
                         static_cast<unsigned long long>(canvas_stats.sum_a));
            }

            if (canvas_stats.nonzero_pixels == 0 && IsPresentTraceEnabled()) {
                InjectStrictCompatDebugOverlay(composed_rgba, layout.width, layout.height);
                const auto overlay_stats = AnalyzeRGBA8Vector(composed_rgba, layout.width, layout.height);
                LOG_INFO(Render_Vulkan,
                         "TRACE_PRESENT cpu_canvas_overlay_injected=1 width={} height={} nonzero={} alpha_nonzero={} opaque={} sum_rgba=({}, {}, {}, {})",
                         layout.width, layout.height,
                         static_cast<unsigned long long>(overlay_stats.nonzero_pixels),
                         static_cast<unsigned long long>(overlay_stats.alpha_nonzero_pixels),
                         static_cast<unsigned long long>(overlay_stats.opaque_pixels),
                         static_cast<unsigned long long>(overlay_stats.sum_r),
                         static_cast<unsigned long long>(overlay_stats.sum_g),
                         static_cast<unsigned long long>(overlay_stats.sum_b),
                         static_cast<unsigned long long>(overlay_stats.sum_a));
            }

            if (layout.width == frame->width && layout.height == frame->height &&
                UploadRGBA8DirectlyToFrameImage(instance, scheduler, frame->image, frame->width,
                                                frame->height, composed_rgba)) {
                if (IsPresentTraceEnabled()) {
                    LOG_INFO(Render_Vulkan,
                             "TRACE_PRESENT cpu_compose_window direct_upload_frame=1 src={}x{} dst={}x{}",
                             layout.width, layout.height, frame->width, frame->height);
                }
                return;
            }

            if (EnsureRGBA8PresentTexture(instance, main_window, screen_infos[0].texture, layout.width,
                                         layout.height) &&
                UploadRGBA8ToImage(instance, scheduler, screen_infos[0].texture.image, layout.width,
                                   layout.height, composed_rgba)) {
                screen_infos[0].image_view = screen_infos[0].texture.image_view;
                screen_infos[0].texcoords = {0.f, 0.f, 1.f, 1.f};

                if (IsPresentTraceEnabled()) {
                    LOG_INFO(Render_Vulkan,
                             "TRACE_PRESENT cpu_compose_window direct_copy_frame=1 src={}x{} dst={}x{}",
                             layout.width, layout.height, frame->width, frame->height);
                }
                DirectCopyPresentTextureToFrame(scheduler, screen_infos[0].texture.image, frame->image,
                                                layout.width, layout.height, frame->width,
                                                frame->height);
                return;
            }
        } else if (IsPresentTraceEnabled()) {
            LOG_INFO(Render_Vulkan,
                     "TRACE_PRESENT cpu_compose_window bypass_gpu_present=1 top_valid={} top_visible={} bottom_valid={} bottom_visible={}",
                     static_cast<u32>(g_strict_compat_cpu_screens[0].valid || g_strict_compat_cpu_screens[1].valid),
                     static_cast<u32>(g_strict_compat_cpu_screens[0].visible || g_strict_compat_cpu_screens[1].visible),
                     static_cast<u32>(g_strict_compat_cpu_screens[2].valid),
                     static_cast<u32>(g_strict_compat_cpu_screens[2].visible));
        }
    }

    PrepareDraw(frame, layout);

    const auto& top_screen = layout.top_screen;
    const auto& bottom_screen = layout.bottom_screen;
    draw_info.modelview = MakeOrthographicMatrix(layout.width, layout.height);

    draw_info.layer = 0;
    if (!Settings::values.swap_screen.GetValue()) {
        DrawTopScreen(layout, top_screen);
        draw_info.layer = 0;
        DrawBottomScreen(layout, bottom_screen);
    } else {
        DrawBottomScreen(layout, bottom_screen);
        draw_info.layer = 0;
        DrawTopScreen(layout, top_screen);
    }

    if (layout.additional_screen_enabled) {
        const auto& additional_screen = layout.additional_screen;
        if (!Settings::values.swap_screen.GetValue()) {
            DrawTopScreen(layout, additional_screen);
        } else {
            DrawBottomScreen(layout, additional_screen);
        }
    }

    scheduler.Record([](vk::CommandBuffer cmdbuf) { cmdbuf.endRenderPass(); });
}


void RendererVulkan::SwapBuffers() {
    const Layout::FramebufferLayout& layout = render_window.GetFramebufferLayout();
    if (IsPresentTraceEnabled()) {
        LOG_INFO(Render_Vulkan, "TRACE_PRESENT swap_buffers begin layout={}x{}", layout.width,
                 layout.height);
    }

    if (IsStrictCompatEnabled()) {
        renderpass_cache.EndRendering();
        main_window.WaitPresent();
        if (second_window) {
            second_window->WaitPresent();
        }
        scheduler.Finish();
        if (IsPresentTraceEnabled()) {
            LOG_INFO(Render_Vulkan,
                     "TRACE_PRESENT strict_compat serialized_before_prepare layout={}x{}",
                     layout.width, layout.height);
        }
    }

    PrepareRendertarget();
    RenderScreenshot();

    if (IsRenderTargetTraceEnabled()) {
        static u64 trace_index = 0;
        const u64 current_trace_index = ++trace_index;
        if (current_trace_index <= GetRenderTargetTraceFrameBudget()) {
            const vk::Device device = instance.GetDevice();
            const u32 width = layout.width;
            const u32 height = layout.height;
            const vk::BufferCreateInfo staging_buffer_info = {
                .size = static_cast<vk::DeviceSize>(width) * static_cast<vk::DeviceSize>(height) * 4,
                .usage = vk::BufferUsageFlagBits::eTransferDst,
            };
            const VmaAllocationCreateInfo alloc_create_info = {
                .flags = VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT |
                         VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT,
                .usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                .requiredFlags = 0,
                .preferredFlags = 0,
                .pool = VK_NULL_HANDLE,
                .pUserData = nullptr,
            };

            VkBuffer unsafe_buffer{};
            VmaAllocation allocation{};
            VmaAllocationInfo alloc_info{};
            VkBufferCreateInfo unsafe_buffer_info = static_cast<VkBufferCreateInfo>(staging_buffer_info);
            const VkResult result = vmaCreateBuffer(instance.GetAllocator(), &unsafe_buffer_info,
                                                    &alloc_create_info, &unsafe_buffer,
                                                    &allocation, &alloc_info);
            if (result != VK_SUCCESS) {
                LOG_INFO(Render_Vulkan, "TRACE_RT staging_alloc_failed result={}", result);
            } else {
                vk::Buffer staging_buffer{unsafe_buffer};
                Frame trace_frame{};
                main_window.RecreateFrame(&trace_frame, width, height);
                DrawScreens(&trace_frame, layout, false);
                scheduler.Record([width, height, source_image = trace_frame.image,
                                  staging_buffer](vk::CommandBuffer cmdbuf) {
                    const vk::ImageMemoryBarrier read_barrier = {
                        .srcAccessMask = vk::AccessFlagBits::eMemoryWrite,
                        .dstAccessMask = vk::AccessFlagBits::eTransferRead,
                        .oldLayout = vk::ImageLayout::eTransferSrcOptimal,
                        .newLayout = vk::ImageLayout::eTransferSrcOptimal,
                        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                        .image = source_image,
                        .subresourceRange{
                            .aspectMask = vk::ImageAspectFlagBits::eColor,
                            .baseMipLevel = 0,
                            .levelCount = VK_REMAINING_MIP_LEVELS,
                            .baseArrayLayer = 0,
                            .layerCount = VK_REMAINING_ARRAY_LAYERS,
                        },
                    };
                    const vk::ImageMemoryBarrier write_barrier = {
                        .srcAccessMask = vk::AccessFlagBits::eTransferRead,
                        .dstAccessMask = vk::AccessFlagBits::eMemoryWrite,
                        .oldLayout = vk::ImageLayout::eTransferSrcOptimal,
                        .newLayout = vk::ImageLayout::eTransferSrcOptimal,
                        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                        .image = source_image,
                        .subresourceRange{
                            .aspectMask = vk::ImageAspectFlagBits::eColor,
                            .baseMipLevel = 0,
                            .levelCount = VK_REMAINING_MIP_LEVELS,
                            .baseArrayLayer = 0,
                            .layerCount = VK_REMAINING_ARRAY_LAYERS,
                        },
                    };
                    static constexpr vk::MemoryBarrier memory_write_barrier = {
                        .srcAccessMask = vk::AccessFlagBits::eMemoryWrite,
                        .dstAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite,
                    };
                    const vk::BufferImageCopy image_copy = {
                        .bufferOffset = 0,
                        .bufferRowLength = 0,
                        .bufferImageHeight = 0,
                        .imageSubresource = {
                            .aspectMask = vk::ImageAspectFlagBits::eColor,
                            .mipLevel = 0,
                            .baseArrayLayer = 0,
                            .layerCount = 1,
                        },
                        .imageOffset = {0, 0, 0},
                        .imageExtent = {width, height, 1},
                    };
                    cmdbuf.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
                                           vk::PipelineStageFlagBits::eTransfer,
                                           vk::DependencyFlagBits::eByRegion, {}, {}, read_barrier);
                    cmdbuf.copyImageToBuffer(source_image, vk::ImageLayout::eTransferSrcOptimal,
                                             staging_buffer, image_copy);
                    cmdbuf.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                                           vk::PipelineStageFlagBits::eAllCommands,
                                           vk::DependencyFlagBits::eByRegion, memory_write_barrier,
                                           {}, write_barrier);
                });
                scheduler.Finish();
                const auto* rgba = static_cast<const u8*>(alloc_info.pMappedData);
                const auto stats = AnalyzeRenderTargetRGBA8(rgba, width, height);
                LOG_INFO(Render_Vulkan,
                         "TRACE_RT main frame={} width={} height={} samples={} nonzero={} alpha_nonzero={} opaque={} sum_rgba=({}, {}, {}, {})",
                         current_trace_index, stats.width, stats.height, stats.sample_count,
                         static_cast<unsigned long long>(stats.nonzero_pixels),
                         static_cast<unsigned long long>(stats.alpha_nonzero_pixels),
                         static_cast<unsigned long long>(stats.opaque_pixels),
                         static_cast<unsigned long long>(stats.sum_r),
                         static_cast<unsigned long long>(stats.sum_g),
                         static_cast<unsigned long long>(stats.sum_b),
                         static_cast<unsigned long long>(stats.sum_a));
                MaybeWriteRenderTargetPPM(rgba, width, height, current_trace_index);
                vmaDestroyBuffer(instance.GetAllocator(), staging_buffer, allocation);
                vmaDestroyImage(instance.GetAllocator(), trace_frame.image, trace_frame.allocation);
                device.destroyFramebuffer(trace_frame.framebuffer);
                device.destroyImageView(trace_frame.image_view);
            }
        }
    }

    RenderToWindow(main_window, layout, false);
#ifndef ANDROID
    if (Settings::values.layout_option.GetValue() == Settings::LayoutOption::SeparateWindows) {
        ASSERT(secondary_window);
        const auto& secondary_layout = secondary_window->GetFramebufferLayout();
        if (!second_window) {
            second_window = std::make_unique<PresentWindow>(*secondary_window, instance, scheduler);
        }
        RenderToWindow(*second_window, secondary_layout, false);
        secondary_window->PollEvents();
    }
#endif
    rasterizer.TickFrame();
    EndFrame();
}

void RendererVulkan::RenderScreenshot() {
    if (!settings.screenshot_requested.exchange(false)) {
        return;
    }

    if (!TryRenderScreenshotWithHostMemory()) {
        RenderScreenshotWithStagingCopy();
    }

    settings.screenshot_complete_callback(false);
}

void RendererVulkan::RenderScreenshotWithStagingCopy() {
    const vk::Device device = instance.GetDevice();

    const Layout::FramebufferLayout layout{settings.screenshot_framebuffer_layout};
    const u32 width = layout.width;
    const u32 height = layout.height;

    const vk::BufferCreateInfo staging_buffer_info = {
        .size = width * height * 4,
        .usage = vk::BufferUsageFlagBits::eTransferDst,
    };

    const VmaAllocationCreateInfo alloc_create_info = {
        .flags = VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT |
                 VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT,
        .usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
        .requiredFlags = 0,
        .preferredFlags = 0,
        .pool = VK_NULL_HANDLE,
        .pUserData = nullptr,
    };

    VkBuffer unsafe_buffer{};
    VmaAllocation allocation{};
    VmaAllocationInfo alloc_info;
    VkBufferCreateInfo unsafe_buffer_info = static_cast<VkBufferCreateInfo>(staging_buffer_info);

    VkResult result = vmaCreateBuffer(instance.GetAllocator(), &unsafe_buffer_info,
                                      &alloc_create_info, &unsafe_buffer, &allocation, &alloc_info);
    if (result != VK_SUCCESS) [[unlikely]] {
        LOG_CRITICAL(Render_Vulkan, "Failed allocating texture with error {}", result);
        UNREACHABLE();
    }

    vk::Buffer staging_buffer{unsafe_buffer};

    Frame frame{};
    main_window.RecreateFrame(&frame, width, height);

    DrawScreens(&frame, layout, false);

    scheduler.Record(
        [width, height, source_image = frame.image, staging_buffer](vk::CommandBuffer cmdbuf) {
            const vk::ImageMemoryBarrier read_barrier = {
                .srcAccessMask = vk::AccessFlagBits::eMemoryWrite,
                .dstAccessMask = vk::AccessFlagBits::eTransferRead,
                .oldLayout = vk::ImageLayout::eTransferSrcOptimal,
                .newLayout = vk::ImageLayout::eTransferSrcOptimal,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = source_image,
                .subresourceRange{
                    .aspectMask = vk::ImageAspectFlagBits::eColor,
                    .baseMipLevel = 0,
                    .levelCount = VK_REMAINING_MIP_LEVELS,
                    .baseArrayLayer = 0,
                    .layerCount = VK_REMAINING_ARRAY_LAYERS,
                },
            };
            const vk::ImageMemoryBarrier write_barrier = {
                .srcAccessMask = vk::AccessFlagBits::eTransferRead,
                .dstAccessMask = vk::AccessFlagBits::eMemoryWrite,
                .oldLayout = vk::ImageLayout::eTransferSrcOptimal,
                .newLayout = vk::ImageLayout::eTransferSrcOptimal,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = source_image,
                .subresourceRange{
                    .aspectMask = vk::ImageAspectFlagBits::eColor,
                    .baseMipLevel = 0,
                    .levelCount = VK_REMAINING_MIP_LEVELS,
                    .baseArrayLayer = 0,
                    .layerCount = VK_REMAINING_ARRAY_LAYERS,
                },
            };
            static constexpr vk::MemoryBarrier memory_write_barrier = {
                .srcAccessMask = vk::AccessFlagBits::eMemoryWrite,
                .dstAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite,
            };

            const vk::BufferImageCopy image_copy = {
                .bufferOffset = 0,
                .bufferRowLength = 0,
                .bufferImageHeight = 0,
                .imageSubresource =
                    {
                        .aspectMask = vk::ImageAspectFlagBits::eColor,
                        .mipLevel = 0,
                        .baseArrayLayer = 0,
                        .layerCount = 1,
                    },
                .imageOffset = {0, 0, 0},
                .imageExtent = {width, height, 1},
            };

            cmdbuf.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
                                   vk::PipelineStageFlagBits::eTransfer,
                                   vk::DependencyFlagBits::eByRegion, {}, {}, read_barrier);
            cmdbuf.copyImageToBuffer(source_image, vk::ImageLayout::eTransferSrcOptimal,
                                     staging_buffer, image_copy);
            cmdbuf.pipelineBarrier(
                vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eAllCommands,
                vk::DependencyFlagBits::eByRegion, memory_write_barrier, {}, write_barrier);
        });

    // Ensure the copy is fully completed before saving the screenshot
    scheduler.Finish();

    // Copy backing image data to the QImage screenshot buffer
    std::memcpy(settings.screenshot_bits, alloc_info.pMappedData, staging_buffer_info.size);

    // Destroy allocated resources
    vmaDestroyBuffer(instance.GetAllocator(), staging_buffer, allocation);
    vmaDestroyImage(instance.GetAllocator(), frame.image, frame.allocation);
    device.destroyFramebuffer(frame.framebuffer);
    device.destroyImageView(frame.image_view);
}

bool RendererVulkan::TryRenderScreenshotWithHostMemory() {
    // If the host-memory import alignment matches the allocation granularity of the platform, then
    // the entire span of memory can be trivially imported
    const bool trivial_import =
        instance.IsExternalMemoryHostSupported() &&
        instance.GetMinImportedHostPointerAlignment() == Common::GetPageSize();
    if (!trivial_import) {
        return false;
    }

    const vk::Device device = instance.GetDevice();

    const Layout::FramebufferLayout layout{settings.screenshot_framebuffer_layout};
    const u32 width = layout.width;
    const u32 height = layout.height;

    // For a span of memory [x, x + s], import [AlignDown(x, alignment), AlignUp(x + s, alignment)]
    // and maintain an offset to the start of the data
    const u64 import_alignment = instance.GetMinImportedHostPointerAlignment();
    const uintptr_t address = reinterpret_cast<uintptr_t>(settings.screenshot_bits);
    void* aligned_pointer = reinterpret_cast<void*>(Common::AlignDown(address, import_alignment));
    const u64 offset = address % import_alignment;
    const u64 aligned_size = Common::AlignUp(offset + width * height * 4ull, import_alignment);

    // Buffer<->Image mapping for the imported imported buffer
    const vk::BufferImageCopy buffer_image_copy = {
        .bufferOffset = offset,
        .bufferRowLength = 0,
        .bufferImageHeight = 0,
        .imageSubresource =
            {
                .aspectMask = vk::ImageAspectFlagBits::eColor,
                .mipLevel = 0,
                .baseArrayLayer = 0,
                .layerCount = 1,
            },
        .imageOffset = {0, 0, 0},
        .imageExtent = {width, height, 1},
    };

    const vk::MemoryHostPointerPropertiesEXT import_properties =
        device.getMemoryHostPointerPropertiesEXT(
            vk::ExternalMemoryHandleTypeFlagBits::eHostAllocationEXT, aligned_pointer);

    if (!import_properties.memoryTypeBits) {
        // Could not import memory
        return false;
    }

    const std::optional<u32> memory_type_index = FindMemoryType(
        instance.GetPhysicalDevice().getMemoryProperties(),
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
        import_properties.memoryTypeBits);

    if (!memory_type_index.has_value()) {
        // Could not find memory type index
        return false;
    }

    const vk::StructureChain<vk::MemoryAllocateInfo, vk::ImportMemoryHostPointerInfoEXT>
        allocation_chain = {
            vk::MemoryAllocateInfo{
                .allocationSize = aligned_size,
                .memoryTypeIndex = memory_type_index.value(),
            },
            vk::ImportMemoryHostPointerInfoEXT{
                .handleType = vk::ExternalMemoryHandleTypeFlagBits::eHostAllocationEXT,
                .pHostPointer = aligned_pointer,
            },
        };

    // Import host memory
    const vk::UniqueDeviceMemory imported_memory =
        device.allocateMemoryUnique(allocation_chain.get());

    const vk::StructureChain<vk::BufferCreateInfo, vk::ExternalMemoryBufferCreateInfo> buffer_info =
        {
            vk::BufferCreateInfo{
                .size = aligned_size,
                .usage = vk::BufferUsageFlagBits::eTransferDst,
                .sharingMode = vk::SharingMode::eExclusive,
            },
            vk::ExternalMemoryBufferCreateInfo{
                .handleTypes = vk::ExternalMemoryHandleTypeFlagBits::eHostAllocationEXT,
            },
        };

    // Bind imported memory to buffer
    const vk::UniqueBuffer imported_buffer = device.createBufferUnique(buffer_info.get());
    device.bindBufferMemory(imported_buffer.get(), imported_memory.get(), 0);

    Frame frame{};
    main_window.RecreateFrame(&frame, width, height);

    DrawScreens(&frame, layout, false);

    scheduler.Record([buffer_image_copy, source_image = frame.image,
                      imported_buffer = imported_buffer.get()](vk::CommandBuffer cmdbuf) {
        const vk::ImageMemoryBarrier read_barrier = {
            .srcAccessMask = vk::AccessFlagBits::eMemoryWrite,
            .dstAccessMask = vk::AccessFlagBits::eTransferRead,
            .oldLayout = vk::ImageLayout::eTransferSrcOptimal,
            .newLayout = vk::ImageLayout::eTransferSrcOptimal,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = source_image,
            .subresourceRange{
                .aspectMask = vk::ImageAspectFlagBits::eColor,
                .baseMipLevel = 0,
                .levelCount = VK_REMAINING_MIP_LEVELS,
                .baseArrayLayer = 0,
                .layerCount = VK_REMAINING_ARRAY_LAYERS,
            },
        };
        const vk::ImageMemoryBarrier write_barrier = {
            .srcAccessMask = vk::AccessFlagBits::eTransferRead,
            .dstAccessMask = vk::AccessFlagBits::eMemoryWrite,
            .oldLayout = vk::ImageLayout::eTransferSrcOptimal,
            .newLayout = vk::ImageLayout::eTransferSrcOptimal,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = source_image,
            .subresourceRange{
                .aspectMask = vk::ImageAspectFlagBits::eColor,
                .baseMipLevel = 0,
                .levelCount = VK_REMAINING_MIP_LEVELS,
                .baseArrayLayer = 0,
                .layerCount = VK_REMAINING_ARRAY_LAYERS,
            },
        };
        static constexpr vk::MemoryBarrier memory_write_barrier = {
            .srcAccessMask = vk::AccessFlagBits::eMemoryWrite,
            .dstAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite,
        };

        cmdbuf.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
                               vk::PipelineStageFlagBits::eTransfer,
                               vk::DependencyFlagBits::eByRegion, {}, {}, read_barrier);
        cmdbuf.copyImageToBuffer(source_image, vk::ImageLayout::eTransferSrcOptimal,
                                 imported_buffer, buffer_image_copy);
        cmdbuf.pipelineBarrier(
            vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eAllCommands,
            vk::DependencyFlagBits::eByRegion, memory_write_barrier, {}, write_barrier);
    });

    // Ensure the copy is fully completed before saving the screenshot
    scheduler.Finish();

    // Image data has been copied directly to host memory
    device.destroyFramebuffer(frame.framebuffer);
    device.destroyImageView(frame.image_view);

    return true;
}

} // namespace Vulkan
