#pragma once
#include <vulkan/vulkan.h>
#include <glm/glm.hpp>
#include "VulkanBuffer.hpp"
class DeviceMemory
{
	
public :
	DeviceMemory(const VkPhysicalDeviceMemoryProperties& memoryProperties, VkDevice device);
	uint32_t getMemoryType(uint32_t typeBits, VkMemoryPropertyFlags properties, VkBool32 *memTypeFound = nullptr);

	VkResult createBuffer(VkBufferUsageFlags usageFlags, VkMemoryPropertyFlags memoryPropertyFlags, Buffer *buffer, VkDeviceSize size, void *data = nullptr);

	uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);

private:

	VkDevice m_device;
	VkPhysicalDeviceMemoryProperties m_memoryProperties;

	

};

