#include "DeviceMemory.h"
#include <stdexcept>



DeviceMemory::DeviceMemory(const VkPhysicalDeviceMemoryProperties& memoryProperties, VkDevice device)
{
	m_memoryProperties = memoryProperties;
	m_device = device;

}





uint32_t DeviceMemory::findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) 
{


	for (uint32_t i = 0; i < m_memoryProperties.memoryTypeCount; i++) {
		if ((typeFilter & (1 << i)) && (m_memoryProperties.memoryTypes[i].propertyFlags & properties) == properties)
		{
			return i;
		}
	}

	throw std::runtime_error("failed to find suitable memory type!");
}

VkResult DeviceMemory::createBuffer(VkBufferUsageFlags usageFlags, VkMemoryPropertyFlags memoryPropertyFlags, Buffer *buffer, VkDeviceSize size, void *data)
{
	buffer->device = m_device;

	VkBufferCreateInfo bufferCreateInfo{};
	bufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	bufferCreateInfo.usage = usageFlags;
	bufferCreateInfo.size = size;

	vkCreateBuffer(m_device, &bufferCreateInfo, nullptr, &buffer->buffer);

	// Create the memory backing up the buffer handle
	VkMemoryRequirements memReqs;

	VkMemoryAllocateInfo memAlloc{};
	memAlloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;

	vkGetBufferMemoryRequirements(m_device, buffer->buffer, &memReqs);
	memAlloc.allocationSize = memReqs.size;
	// Find a memory type index that fits the properties of the buffer
	memAlloc.memoryTypeIndex = getMemoryType(memReqs.memoryTypeBits, memoryPropertyFlags);
	vkAllocateMemory(m_device, &memAlloc, nullptr, &buffer->memory);

	buffer->alignment = memReqs.alignment;
	buffer->size = size;
	buffer->usageFlags = usageFlags;
	buffer->memoryPropertyFlags = memoryPropertyFlags;

	// If a pointer to the buffer data has been passed, map the buffer and copy over the data
	if (data != nullptr)
	{
		buffer->map();
		memcpy(buffer->mapped, data, size);
		if ((memoryPropertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) == 0)
			buffer->flush();

		buffer->unmap();
	}

	// Initialize a default descriptor that covers the whole buffer size
	buffer->setupDescriptor();

	// Attach the memory to the buffer object
	return buffer->bind();
}

uint32_t DeviceMemory::getMemoryType(uint32_t typeBits, VkMemoryPropertyFlags properties, VkBool32 *memTypeFound)
{

	for (uint32_t i = 0; i < m_memoryProperties.memoryTypeCount; i++)
	{
		if ((typeBits & 1) == 1)
		{
			if ((m_memoryProperties.memoryTypes[i].propertyFlags & properties) == properties)
			{
				if (memTypeFound)
				{
					*memTypeFound = true;
				}
				return i;
			}
		}
		typeBits >>= 1;
	}

	if (memTypeFound)
	{
		*memTypeFound = false;
		return 0;
	}
	else
	{
		throw std::runtime_error("Could not find a matching memory type");
	}
}
