#include <cstdio>
#include "Image.hpp"
#include "Composer.cpp"

int main(void){

	ImageNS::BitMap cat("Pictures/Cat.bmp");
	ImageNS::BitMap table("Pictures/Table.bmp");

	size_t cat_offset = 0;

	//for (int i = 0; i < 50000; ++i){
		compose(cat.pixels(), cat.width(), cat.height(), table.pixels() + cat_offset, table.width(), table.height());
	//}

	table.write("merged.bmp");

	return 0;
}
