#include <cstdint>
#include <cstdlib>
#include <immintrin.h>

void compose(char* front, size_t f_width, size_t f_height, char* back, size_t b_width, size_t b_height){


	const __m256i zeroes         = _mm256_set_epi8 (0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 
	                                                0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
	                                                0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 
	                                                0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00); 

	const __m256i alpha_mask     = _mm256_set_epi8 (0xFF, 0x1E, 0xFF, 0x1E, 0xFF, 0x1E, 0xFF, 0x1E,
	                                                0xFF, 0x16, 0xFF, 0x16, 0xFF, 0x16, 0xFF, 0x16, 
	                                                0xFF, 0x0E, 0xFF, 0x0E, 0xFF, 0x0E, 0xFF, 0x0E,
	                                                0xFF, 0x06, 0xFF, 0x06, 0xFF, 0x06, 0xFF, 0x06);

	const __m256i OxFF_mask      = _mm256_set_epi8 (0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF,
	                                                0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF,
	                                                0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF,
	                                                0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF);

	const __m256i extract_mask_h = _mm256_set_epi8 (0x1E, 0x1C, 0x1A, 0x18, 0x16, 0x14, 0x12, 0x10,
	                                                0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
	                                                0x0E, 0x0C, 0x0A, 0x08, 0x06, 0x04, 0x02, 0x00,
	                                                0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF);

	const __m256i extract_mask_l = _mm256_set_epi8 (0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
	                                                0x0E, 0x0C, 0x0A, 0x08, 0x06, 0x04, 0x02, 0x00,
	                                                0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
	                                                0x1E, 0x1C, 0x1A, 0x18, 0x16, 0x14, 0x12, 0x10);


	const int PIXEL_SIZE = 4;

	for (size_t y = 0; y < f_height; ++y){
		for (size_t x = 0; x < f_width; x += 8){

			__m256i bp   = _mm256_loadu_si256(reinterpret_cast<__m256i*>(back  + x * PIXEL_SIZE)); //vmovdqa
			__m256i fp   = _mm256_loadu_si256(reinterpret_cast<__m256i*>(front + x * PIXEL_SIZE)); //vmocdqa

			__m256i bp_h =  _mm256_unpackhi_epi8(bp, zeroes);
			__m256i bp_l =  _mm256_unpacklo_epi8(bp, zeroes);

			__m256i fp_h =  _mm256_unpackhi_epi8(fp, zeroes);
			__m256i fp_l =  _mm256_unpacklo_epi8(fp, zeroes);

//Shuffle pixel to extract alpha
			__m256i fpa_l = _mm256_shuffle_epi8(fp_l, alpha_mask);
			__m256i fpa_h = _mm256_shuffle_epi8(fp_h, alpha_mask);

//  RES = (F * α) + (B * (255 - α))
//-----------------------------------
//               255
//               
			__m256i nbp_h    = _mm256_mullo_epi16(bp_h, _mm256_sub_epi16(OxFF_mask, fpa_h));
			__m256i nbp_l    = _mm256_mullo_epi16(bp_l, _mm256_sub_epi16(OxFF_mask, fpa_l));

			__m256i nfp_h    = _mm256_mullo_epi16(fp_h, fpa_h);
			__m256i nfp_l    = _mm256_mullo_epi16(fp_l, fpa_l);

//Division is replaced by shift for better performance 
			__m256i res_h    = _mm256_srli_epi16(_mm256_add_epi16(nbp_h, nfp_h), 8);
			__m256i res_l    = _mm256_srli_epi16(_mm256_add_epi16(nbp_l, nfp_l), 8);

			res_h = _mm256_shuffle_epi8(res_h, extract_mask_h);
			res_l = _mm256_shuffle_epi8(res_l, extract_mask_l);

			_mm256_storeu_si256(reinterpret_cast<__m256i*>(back + PIXEL_SIZE * x), _mm256_or_si256(res_l, res_h));
		}

		front += 4 * f_width;
		back  += 4 * b_width;
	}
}

void slow_compose(char* front, size_t f_width, size_t f_height, char* back, size_t b_width, size_t b_height){

	const int PIXEL_SIZE = 4;

	for (size_t y = 0; y < f_height; ++y){
		for (size_t x = 0; x < f_width; x += 1){

			uint8_t alpha = *(uint8_t*)(front + PIXEL_SIZE * x + 3);
			uint8_t inv_alpha = 255 - alpha;

			back[PIXEL_SIZE * x]     = (uint8_t)((alpha * front[4 * x]     + inv_alpha * back[4 * x])     >> 8);
			back[PIXEL_SIZE * x + 1] = (uint8_t)((alpha * front[4 * x + 1] + inv_alpha * back[4 * x + 1]) >> 8);
			back[PIXEL_SIZE * x + 2] = (uint8_t)((alpha * front[4 * x + 2] + inv_alpha * back[4 * x + 2]) >> 8);

		}

	front += PIXEL_SIZE * f_width;
	back  += PIXEL_SIZE * b_width;
	}
}

