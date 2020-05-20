#include <cstdint>
#include <cstdlib>
#include <immintrin.h>

void compose(char* front, size_t f_width, size_t f_height, char* back, size_t b_width, size_t b_height){


	const __m512i zeroes         = _mm512_set_epi8 (0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 
	                                                0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
	                                                0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 
	                                                0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 
	                                                0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 
	                                                0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 
	                                                0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 
	                                                0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00);

	const __m512i alpha_mask     = _mm512_set_epi8 (0xFF, 0x3E, 0xFF, 0x3E, 0xFF, 0x3E, 0xFF, 0x3E,
	                                                0xFF, 0x36, 0xFF, 0x36, 0xFF, 0x36, 0xFF, 0x36, 
	                                                0xFF, 0x2E, 0xFF, 0x2E, 0xFF, 0x2E, 0xFF, 0x2E,
	                                                0xFF, 0x26, 0xFF, 0x26, 0xFF, 0x26, 0xFF, 0x26, 
	                                                0xFF, 0x1E, 0xFF, 0x1E, 0xFF, 0x1E, 0xFF, 0x1E,
	                                                0xFF, 0x16, 0xFF, 0x16, 0xFF, 0x16, 0xFF, 0x16, 
	                                                0xFF, 0x0E, 0xFF, 0x0E, 0xFF, 0x0E, 0xFF, 0x0E,
	                                                0xFF, 0x06, 0xFF, 0x06, 0xFF, 0x06, 0xFF, 0x06);

	const __m512i OxFF_mask      = _mm512_set_epi8 (0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF,
	                                                0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF,
	                                                0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF,
	                                                0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF,
	                                                0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF,
	                                                0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF,
	                                                0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF,
	                                                0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF);

	const __m512i extract_mask_h = _mm512_set_epi8 (0x3E, 0x3C, 0x3A, 0x38, 0x36, 0x34, 0x32, 0x30,
	                                                0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
	                                                0x2E, 0x0C, 0x2A, 0x28, 0x26, 0x24, 0x22, 0x20,
	                                                0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
	                                                0x1E, 0x1C, 0x1A, 0x18, 0x16, 0x14, 0x12, 0x10,
	                                                0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
	                                                0x0E, 0x0C, 0x0A, 0x08, 0x06, 0x04, 0x02, 0x00,
	                                                0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF);

	const __m512i extract_mask_l = _mm512_set_epi8 (0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
	                                                0x0E, 0x0C, 0x0A, 0x08, 0x06, 0x04, 0x02, 0x00,
	                                                0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
	                                                0x1E, 0x1C, 0x1A, 0x18, 0x16, 0x14, 0x12, 0x10,
			                                        0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
	                                                0x2E, 0x2C, 0x2A, 0x28, 0x26, 0x24, 0x22, 0x20,
	                                                0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
	                                                0x3E, 0x3C, 0x3A, 0x38, 0x36, 0x34, 0x32, 0x30);


	const int PIXEL_SIZE = 4;

	for (size_t y = 0; y < f_height; ++y){
		for (size_t x = 0; x < f_width; x += 16){

			__m512i bp   = _mm512_loadu_si512(reinterpret_cast<__m512i*>(back  + x * PIXEL_SIZE)); //vmovdqa
			__m512i fp   = _mm512_loadu_si512(reinterpret_cast<__m512i*>(front + x * PIXEL_SIZE)); //vmocdqa

			__m512i bp_h =  _mm512_unpackhi_epi8(bp, zeroes);
			__m512i bp_l =  _mm512_unpacklo_epi8(bp, zeroes);

			__m512i fp_h =  _mm512_unpackhi_epi8(fp, zeroes);
			__m512i fp_l =  _mm512_unpacklo_epi8(fp, zeroes);

//Shuffle pixel to extract alpha
			__m512i fpa_l = _mm512_shuffle_epi8(fp_l, alpha_mask);
			__m512i fpa_h = _mm512_shuffle_epi8(fp_h, alpha_mask);

//  RES = (F * α) + (B * (255 - α))
//-----------------------------------
//               255
//               
			__m512i nbp_h    = _mm512_mullo_epi16(bp_h, _mm512_sub_epi16(OxFF_mask, fpa_h));
			__m512i nbp_l    = _mm512_mullo_epi16(bp_l, _mm512_sub_epi16(OxFF_mask, fpa_l));

			__m512i nfp_h    = _mm512_mullo_epi16(fp_h, fpa_h);
			__m512i nfp_l    = _mm512_mullo_epi16(fp_l, fpa_l);

//Division is replaced by shift for better performance 
			__m512i res_h    = _mm512_srli_epi16(_mm512_add_epi16(nbp_h, nfp_h), 8);
			__m512i res_l    = _mm512_srli_epi16(_mm512_add_epi16(nbp_l, nfp_l), 8);

			res_h = _mm512_shuffle_epi8(res_h, extract_mask_h);
			res_l = _mm512_shuffle_epi8(res_l, extract_mask_l);

			_mm512_storeu_si512(reinterpret_cast<__m512i*>(back + PIXEL_SIZE * x), _mm512_or_si512(res_l, res_h));
		}

		front += 4 * f_width;
		back  += 4 * b_width;
	}
}