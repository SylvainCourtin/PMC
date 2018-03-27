#include <vector>
#include <chrono>
#include <iostream>

#include <algorithm>
#include <immintrin.h>
#include <boost/align/aligned_allocator.hpp>

#define LENGTH 512
#define LENGTH_TEST 32
#define LT2 6

static float g = 1.0f;
using vec16 = std::vector<float,boost::alignment::aligned_allocator<float,LENGTH_TEST> >;

inline __m256 shift1R(__m256 a, __m256 b){
	auto couple = _mm256_permute2f128_ps(a,b,0x21);
	auto blend = _mm256_blend_ps (a, couple, 0x11);
	auto permute = _mm256_permute_ps(blend,0x39);	
	return permute;
}

inline __m256 shift2R(__m256 a, __m256 b){
	auto couple = _mm256_permute2f128_ps(a,b,0x21);
	auto shuffle = _mm256_shuffle_ps(a,couple,0x4E);
	return shuffle;
}

inline __m256 shift3R(__m256 a, __m256 b){
	auto couple = _mm256_permute2f128_ps(a,b,0x21);
	auto blend = _mm256_blend_ps (a, couple, 0x77);
	auto permute = _mm256_permute_ps(blend,0x93);

	return permute;
}

inline void storeShift1L(vec16 &tab, __m256 a, __m256 b, __m256 soluce){
	
	auto permuteG = _mm256_permute_ps(soluce,0x93);
	auto permute2f128 = _mm256_permute2f128_ps(permuteG,permuteG,0x01);
	auto blend1 = _mm256_blend_ps(permuteG, a, 0x01);
	auto blend1_2 = _mm256_blend_ps (blend1, permute2f128, 0x10);
	
	auto blend2 = _mm256_blend_ps (b, permute2f128, 0x01);
	
	_mm256_store_ps(&tab[LENGTH_TEST],blend1_2);
	_mm256_store_ps(&tab[LENGTH_TEST+8],blend2);
	
	std::cout << "a :\t";
	for(int i = 0; i < 8; i++){
		std::cout << a[i] << "\t";
	}
	std::cout << std::endl;
	
	std::cout << "b :\t";
	for(int i = 0; i < 8; i++){
		std::cout << b[i] << "\t";
	}
	std::cout << std::endl;
	
	std::cout << "s :\t";
	for(int i = 0; i < 8; i++){
		std::cout << soluce[i] << "\t";
	}
	std::cout << std::endl;
	
	std::cout << "pg :\t";
	for(int i = 0; i < 8; i++){
		std::cout << permuteG[i] << "\t";
	}
	std::cout << std::endl;
	
	std::cout << "p2 :\t";
	for(int i = 0; i < 8; i++){
		std::cout << permute2f128[i] << "\t";
	}
	std::cout << std::endl;
	
	std::cout << "b1 :\t";
	for(int i = 0; i < 8; i++){
		std::cout << blend1[i] << "\t";
	}
	std::cout << std::endl;
	
	std::cout << "b12 :\t";
	for(int i = 0; i < 8; i++){
		std::cout << blend1_2[i] << "\t";
	}
	std::cout << std::endl;
	
	std::cout << "b2 :\t";
	for(int i = 0; i < 8; i++){
		std::cout << blend2[i] << "\t";
	}
	std::cout << std::endl;
}

inline __m256 storeShift1LFor(vec16 &tab, __m256 ancien, __m256 b, __m256 soluce, size_t i, size_t j){
	
	auto permuteG = _mm256_permute_ps(soluce,0x93);
	auto permute2f128 = _mm256_permute2f128_ps(permuteG,permuteG,0x01);
	auto blend1 = _mm256_blend_ps(permuteG, ancien, 0x01);
	auto blend1_2 = _mm256_blend_ps (blend1, permute2f128, 0x10);
	
	auto blend2 = _mm256_blend_ps (b, permute2f128, 0x01);
	
	_mm256_store_ps(&tab[i*LENGTH_TEST+j],blend1_2);
	//_mm256_store_ps(&tab[i*LENGTH_TEST+(j+8)],blend2);
	
	return blend2;
}

inline void storeShift2L(vec16 &tab, __m256 a, __m256 b, __m256 soluce){
	
	auto permuteG2 = _mm256_permute_ps(soluce,0x4E);
	auto permute2f128 = _mm256_permute2f128_ps(permuteG2,permuteG2,0x01);
	auto blend1 = _mm256_blend_ps(permuteG2, a, 0x03);
	auto blend1_2 = _mm256_blend_ps (blend1, permute2f128, 0x30);
	
	auto blend2 = _mm256_blend_ps (b, permute2f128, 0x03);
	
	_mm256_store_ps(&tab[LENGTH_TEST],blend1_2);
	_mm256_store_ps(&tab[LENGTH_TEST+8],blend2);
}

inline __m256 add(__m256 h, __m256 g, __m256 c, __m256 d, __m256 b){
	auto stencil = _mm256_add_ps(h,g);
	stencil = _mm256_add_ps(stencil,c);
	stencil = _mm256_add_ps(stencil,d);
	stencil = _mm256_add_ps(stencil,b);
	return stencil;
}  



void version1(){
	vec16 tab(LENGTH_TEST*3);
	std::generate(tab.begin(), tab.end(), [](){int tmp=g++; return (tmp-1)%LENGTH_TEST;});
	
	for_each(tab.begin(), tab.end(), [](float t){std::cout << t << "\t";});
	std::cout << "\n" << std::endl;
	
	auto l1_1 = _mm256_load_ps(&tab[0]);
	auto l1_2 = _mm256_load_ps(&tab[0+8]);
	auto l1d1 = shift1R(l1_1,l1_2);
	
	auto l2 = _mm256_load_ps(&tab[0+LENGTH_TEST]);
	auto l2_2 = _mm256_load_ps(&tab[0+8+LENGTH_TEST]);
	
	auto l2d1 = shift1R(l2,l2_2);
	auto l2d2 = shift2R(l2,l2_2);
	
	auto l3_1 = _mm256_load_ps(&tab[0+LENGTH_TEST*2]);
	auto l3_2 = _mm256_load_ps(&tab[0+8+LENGTH_TEST*2]);
	auto l3d1 = shift1R(l3_1,l3_2);
	
	auto l2a1 = add(l1d1,l2,l2d1,l2d2,l3d1);
	
	/*auto l2a1 = _mm256_add_ps(l1d1,l2);
	l2a1 = _mm256_add_ps(l2a1,l2d1);
	l2a1 = _mm256_add_ps(l2a1,l2d2);
	l2a1 = _mm256_add_ps(l2a1,l3d1);*/
	
	//_mm256_storeu_ps(&tab[LENGTH_TEST+1],l2a1);
	
	
	storeShift1L(tab, l2, l2_2, l2a1); 
	//storeShift2L(tab, l2, l2_2, l2a1); 	

	for_each(tab.begin(), tab.end(), [](float t){std::cout << t << "\t";});
	std::cout << "\n" << std::endl;
	
}

void version2(){
	//std::vector<float> tab(LENGTH_TEST*LT2,1.0f);
	//std::vector<float> tab_tmp(LENGTH_TEST*LT2,0.0f);
	//std::vector<float> fin(LENGTH_TEST*LT2);
	
	vec16 tab(LENGTH_TEST*LT2,1.0f);
	vec16 tab_tmp(LENGTH_TEST*LT2,0.0f);
	vec16 fin(LENGTH_TEST*LT2);
	
	fin = tab;
	
	//tab_tmp = tab;
	
	size_t sauvj = 0;
	
	for(size_t i = 1; i < LT2-1; i++){
		auto l1_1 = _mm256_load_ps(&tab[(i-1)*LENGTH_TEST]);
		auto l1_2 = _mm256_load_ps(&tab[(i-1)*LENGTH_TEST+8]);
		auto l1d1 = shift1R(l1_1,l1_2);
	
		auto l2 = _mm256_load_ps(&tab[i*LENGTH_TEST]);
		auto l2_2 = _mm256_load_ps(&tab[i*LENGTH_TEST+8]);
		auto l2d1 = shift1R(l2,l2_2);
		auto l2d2 = shift2R(l2,l2_2);
	
		auto l3_1 = _mm256_load_ps(&tab[(i+1)*LENGTH_TEST]);
		auto l3_2 = _mm256_load_ps(&tab[(i+1)*LENGTH_TEST+8]);
		auto l3d1 = shift1R(l3_1,l3_2);
	
		auto l2a1 = add(l1d1,l2,l2d1,l2d2,l3d1);
		
		auto ancien = storeShift1LFor(tab_tmp, l2, l2_2, l2a1, i, 0); 
		/////////////////////////////////////////////////
		
		std::cout << "j : 0" << std::endl;
		
		std::cout << "\n";
		for(int i = 0; i < LT2; i++){
			for(int j = 0; j < LENGTH_TEST; j++){
				std::cout << tab_tmp[i*LENGTH_TEST+j];
			}
			std::cout << "\n";
		}
		std::cout << "\n" << std::endl;
		/////////////////////////////////////////////////
		for(size_t j = 8; j < LENGTH_TEST-8; j+=8){
			
			std::cout << "i/j : " << i << "/" << j << std::endl;
			
			l1_1 = l1_2;
			l2 = l2_2;
			l3_1 = l3_2;
			
			auto l1_2 = _mm256_load_ps(&tab[(i-1)*LENGTH_TEST + (j+8)]);
			auto l2_2 = _mm256_load_ps(&tab[i*LENGTH_TEST + (j+8)]);
			auto l3_2 = _mm256_load_ps(&tab[(i+1)*LENGTH_TEST + (j+8)]);
			auto l1d1 = shift1R(l1_1,l1_2);
			auto l2d1 = shift1R(l2,l2_2);
			auto l2d2 = shift2R(l2,l2_2);
			auto l3d1 = shift1R(l3_1,l3_2);

			auto l2a1 = add(l1d1,l2,l2d1,l2d2,l3d1);
			
			ancien = storeShift1LFor(tab_tmp, ancien, l2_2, l2a1, i, j);
			/////////////////////////////////////////////////
		
			std::cout << "\n";
			for(int i = 0; i < LT2; i++){
				for(int j = 0; j < LENGTH_TEST; j++){
					std::cout << tab_tmp[i*LENGTH_TEST+j];
				}
				std::cout << "\n";
			}
			std::cout << "\n" << std::endl;
		
		/////////////////////////////////////////////////
		 
		 sauvj = j+8;
		}
		
		//ou récupération du premier de ancien et copier coller comme la fin de boucle
		_mm256_store_ps(&tab_tmp[i*LENGTH_TEST+sauvj],ancien);
		
		
		std::cout << "\n";
			for(int i = 0; i < LT2; i++){
				for(int j = 0; j < LENGTH_TEST; j++){
					std::cout << tab_tmp[i*LENGTH_TEST+j];
				}
				std::cout << "\n";
			}
			std::cout << "\n" << std::endl;
		
		for(std::size_t j=sauvj+1;j<LENGTH_TEST-1;j++)
			{
				float h(0),b(0),g(0),d(0);
				
				h=tab[(i)* LENGTH_TEST+j-1];
				b=tab[(i)* LENGTH_TEST+j+1];
				g=tab[(i-1)* LENGTH_TEST+j];
				d=tab[(i+1)* LENGTH_TEST+j];
				
				float tmp = h*h+b*b+g*g+d*d;
				
				tab_tmp[i*LENGTH_TEST+j] = (tab[i*LENGTH_TEST+j] + tmp) ;
			}
	}
	
	tab=tab_tmp;
	
	//for_each(tab.begin(), tab.end(), [](float t){std::cout << t << "\t";});
	std::cout << "\n";
	for(int i = 0; i < LT2; i++){
		for(int j = 0; j < LENGTH_TEST; j++){
			std::cout << tab[i*LENGTH_TEST+j];
		}
		std::cout << "\n";
	}
	std::cout << "\n" << std::endl;
	
	//auto a = _mm256_set1_ps(x[is0 + k]);
	//auto b = _mm256_set1_ps(x[is0 + k + 1]);
}

int main(){
	version2();
	
	/*std::cout << "a :\t";
	for(int i = 0; i < 8; i++){
		std::cout << a[i] << "\t";
	}
	std::cout << std::endl;
	
	std::cout << "b :\t";
	for(int i = 0; i < 8; i++){
		std::cout << b[i] << "\t";
	}
	std::cout << std::endl;
	
	std::cout << "c :\t";
	for(int i = 0; i < 8; i++){
		std::cout << couple[i] << "\t";
	}
	std::cout << std::endl;
	
	std::cout << "shu :\t";
	for(int i = 0; i < 8; i++){
		std::cout << shuffle[i] << "\t";
	}
	std::cout << std::endl;*/
	
	
	return 0;
}