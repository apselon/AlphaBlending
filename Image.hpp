#pragma once
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <stdexcept>

namespace ImageNS {

	const size_t DATA_OFFSET   = 10;
	const size_t WIDTH_OFFSET  = 18;
	const size_t HEIGHT_OFFSET = 22;

	const uint16_t BM_LABEL = 0x424d;
	
	class BitMap {
	private:
		const char* filename = "Unnamed.bmp";

		const char** data = nullptr;
	
		size_t width  = 0;
		size_t height = 0;

	
	public:
		BitMap() = default;
		explicit BitMap(const char* filename);

		~BitMap();
	};

	BitMap::BitMap(const char* filename): filename(filename){

		FILE* image_f = fopen("filename", "r");

		uint16_t label = 0;
		fread(&label, sizeof(uint16_t), 1, image_f);

		if (label != BM_LABEL) throw std::invalid_argument("Wrong file format. Expcepted BMP");

	}

};
