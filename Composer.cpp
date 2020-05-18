#include <cstdint>
#include <cstdlib>
#include <immintrin.h>

void compose(char* front, size_t f_width, size_t f_height, char* back, size_t b_width, size_t b_height){

	const __m128i zeroes         = _mm_set_epi8 (0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
	                                             0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00);

	const __m128i alpha_mask     = _mm_set_epi8 (0xFF, 0x0E, 0xFF, 0x0E, 0xFF, 0x0E, 0xFF, 0x0E,
	                                             0xFF, 0x06, 0xFF, 0x06, 0xFF, 0x06, 0xFF, 0x06);

	const __m128i OxFF_mask      = _mm_set_epi8 (0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF,
	                                             0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF);

	const __m128i extract_mask_h = _mm_set_epi8 (0x0E, 0x0C, 0x0A, 0x08, 0x06, 0x04, 0x02, 0x00,
	                                             0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF);

	const __m128i extract_mask_l = _mm_set_epi8 (0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
	                                             0x0E, 0x0C, 0x0A, 0x08, 0x06, 0x04, 0x02, 0x00);

	const int PIXEL_SIZE = 4;

	for (size_t y = 0; y < f_height; ++y){
		for (size_t x = 0; x < f_width; x += PIXEL_SIZE){

			__m128i bp   = _mm_loadu_si128(reinterpret_cast<__m128i*>(back  + x * PIXEL_SIZE));
			__m128i fp   = _mm_loadu_si128(reinterpret_cast<__m128i*>(front + x * PIXEL_SIZE));

			__m128i bp_h = reinterpret_cast<__m128i>(_mm_movehl_ps(reinterpret_cast<__m128>(zeroes),
			                                                       reinterpret_cast<__m128>(bp    )));

			__m128i bp_l = reinterpret_cast<__m128i>(_mm_movelh_ps(reinterpret_cast<__m128>(bp    ),
			                                                       reinterpret_cast<__m128>(zeroes)));


			__m128i fp_h = reinterpret_cast<__m128i>(_mm_movehl_ps(reinterpret_cast<__m128>(zeroes),
			                                                       reinterpret_cast<__m128>(fp    )));

			__m128i fp_l = reinterpret_cast<__m128i>(_mm_movelh_ps(reinterpret_cast<__m128>(fp    ),
			                                                       reinterpret_cast<__m128>(zeroes)));

//Zero extend packed unsigned 8-bit integers in a to packed 16-bit integers, and store the results in dst.
			__m128i bp_h_ext = _mm_cvtepi8_epi16(bp_h);
			__m128i bp_l_ext = _mm_cvtepi8_epi16(bp_l);

			__m128i fp_h_ext = _mm_cvtepi8_epi16(fp_h);
			__m128i fp_l_ext = _mm_cvtepi8_epi16(fp_l);

//Shuffle pixel to extract alpha
			__m128i fpa_h    = _mm_shuffle_epi8(fp_h_ext, alpha_mask);
			__m128i fpa_l    = _mm_shuffle_epi8(fp_l_ext, alpha_mask);
			
//  RES = (F * α) + (B * (255 - α))
//-----------------------------------
//               255
			__m128i nbp_h    = _mm_mullo_epi16(bp_h_ext, _mm_sub_epi16(OxFF_mask, fpa_h));
			__m128i nbp_l    = _mm_mullo_epi16(bp_l_ext, _mm_sub_epi16(OxFF_mask, fpa_l));

			__m128i nfp_h    = _mm_mullo_epi16(fp_h_ext, fpa_h);
			__m128i nfp_l    = _mm_mullo_epi16(fp_l_ext, fpa_l);

//Division is replaced by shift for better performance 
			__m128i res_h    = _mm_srli_epi16(_mm_add_epi16(nbp_h, nfp_h), 8);
			__m128i res_l    = _mm_srli_epi16(_mm_add_epi16(nbp_l, nfp_l), 8);

			res_h = _mm_shuffle_epi8(res_h, extract_mask_h);
			res_l = _mm_shuffle_epi8(res_l, extract_mask_l);

			_mm_storeu_si128(reinterpret_cast<__m128i*>(back + PIXEL_SIZE * x), _mm_or_si128(res_l, res_h));
		}

		front += PIXEL_SIZE * f_width;
		back  += PIXEL_SIZE * b_width;
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

