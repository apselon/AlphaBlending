#pragma once
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <stdexcept>

#define NONCOPYABLE(classname)                              \
    classname           (const classname& second) = delete; \
    classname& operator=(const classname& second) = delete; \
                                                            \

namespace ImageNS {

	const size_t PIXEL_SIZE         =  4;
	const size_t WIDTH_OFFSET       = 18;
	const size_t HEIGHT_OFFSET      = 22;
	const size_t DATA_OFFSET_OFFSET = 10;

	const uint16_t BM_LABEL = 0x4d42;
	

	class BitMap {

	private:
		char* data_   __attribute__((aligned(16)))  = nullptr; 
		char* pixels_ __attribute__((aligned(16)))  = nullptr;

		uint32_t width_  = 0;
		uint32_t height_ = 0;
		uint32_t data__offset = 0;
		uint32_t size = 0;

	public:
		char* pixels(){
			return this->pixels_;
		}

		uint32_t width(){
			return this->width_;
		}

		uint32_t height(){
			return this->height_;
		}

		NONCOPYABLE(BitMap);

		BitMap() = default;
		explicit BitMap(const char* filename);
		void write(const char* filename);
		~BitMap();
	};

	BitMap::BitMap(const char* filename){

		FILE* image_f = fopen(filename, "rb");

		fseek(image_f, 0L, SEEK_END);
		size = ftell(image_f);
		rewind(image_f);

		data_ = new char[size + 128];

		fread(data_, sizeof(char), size, image_f);

		data__offset = *(uint32_t*)(data_ + DATA_OFFSET_OFFSET);
		pixels_ = data_ + data__offset;
		width_  =  *(uint32_t*)(data_ + WIDTH_OFFSET);
		height_ =  *(uint32_t*)(data_ + HEIGHT_OFFSET);

		fclose(image_f);
	}

	void BitMap::write(const char* filename){
		FILE* image_f = fopen(filename, "wb");
		fwrite(data_, sizeof(char), size, image_f);
		fclose(image_f);
	}

	BitMap::~BitMap(){
		delete [] data_;
	}

};
