#include <vector>
#include <chrono>
#include <iostream>

#include <boost/align/aligned_allocator.hpp>
#include <immintrin.h>

using vec = std::vector<float,boost::alignment::aligned_allocator<float,32> >;

static std::size_t const N=512;
static std::size_t const N2=N*N;

//Version naïf
/**
 * Temps = ~226ms avec -O3 | ~4829ms sans -O3
 * 
 * */
void version_scalaire()
{
	std::vector<float> v(N*N*N,1.0f);
	std::vector<float> v_tmp(N*N*N);
	
	v_tmp = v;
	
	auto start = std::chrono::system_clock::now();
		#pragma omp parallel for
		for(std::size_t i=1;i<N-1; i++)
		{
			for(std::size_t j=1;j<N-1; j++)
			{
				for(std::size_t k=1; k<N-1; k++)
				{
					float x1,x2,y1,y2,z1,z2;
					x1=v[(i+1)*N2+j*N+k];
					x2=v[(i-1)*N2+j*N+k];
					
					y1=v[i*N2+(j+1)*N+k];
					y2=v[i*N2+(j-1)*N+k];
					
					z1=v[i*N2+j*N+(k+1)];
					z2=v[i*N2+j*N+(k-1)];
					
					float tmp = x1*x1+x2*x2+y1*y1+y2*y2+z1*z1+z2*z2;
					
					v_tmp[i*N2+j*N+k] = (v[i*N2+j*N+k]*v[i*N2+j*N+k]  + tmp);
				}
				
			}
			
		}
	v = v_tmp;
	auto stop = std::chrono::system_clock::now();
	std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(stop-start).count() << "ms"<< std::endl;	
	
	/*for(std::size_t i=0;i<N; ++i)
	{
		for(std::size_t j=0;j<N; j++)
		{
			for(std::size_t k=0; k<N; k++)
			{
				std::cout << v[i*N2+j*N+k] << " ";
			}
			std::cout << std::endl;
		}
		std::cout << "\n----------------------------------\n i="<< i << std::endl;
	}*/
}

inline __m256 shift1(__m256 a, __m256 b)
{
	auto c = _mm256_permute2f128_ps(a,b,0x21);
	auto d = _mm256_blend_ps(a,c,0x11);
	return _mm256_permute_ps(d,0x39);
}


inline __m256 shift2(__m256 a, __m256 b)
{
	auto c = _mm256_permute2f128_ps(a,b,0x21);
	return _mm256_shuffle_ps(a,c,0x4E);
}

inline void shift1Reverse(__m256 &a, __m256 &b, __m256 soluce)
{
	auto permute = _mm256_permute_ps(soluce,0x93);
	auto permute2f128 = _mm256_permute2f128_ps(permute,permute,0x01);
	auto blend1 = _mm256_blend_ps(permute,a,0x01);
	auto blendbis = _mm256_blend_ps(blend1,permute2f128,0x10);
	a = blendbis;
	
	auto blend2 = _mm256_blend_ps(b, permute2f128, 0x01);
	b = blend2;	
}

/**
 * Pour la partie store
 * Actuellement même avec le store perso il est plus lent que le store U en terme de temps :( mais on a essayé
 * 
 * */
 
 inline __m256 storeShift1LFor3D(vec &v_tmp, __m256 a, __m256 b, __m256 soluce, size_t i, size_t j, size_t k){
	
	auto permuteG = _mm256_permute_ps(soluce,0x93);
	auto permute2f128 = _mm256_permute2f128_ps(permuteG,permuteG,0x01);
	auto blend1 = _mm256_blend_ps(permuteG, a, 0x01);
	auto blend1_2 = _mm256_blend_ps (blend1, permute2f128, 0x10);
	
	auto blend2 = _mm256_blend_ps (b, permute2f128, 0x01);
	
	_mm256_store_ps(&v_tmp[i*N2+j*N+k],blend1_2);
	_mm256_store_ps(&v_tmp[i*N+j*N+(k+8)],blend2);
	
	return blend2;
}


/**
 * temps 246ms | 1354 ms sans -O3
 * */
void version_mm256_avecLoadU_avecStoreU()
{
	vec v(N*N*N,1.0f);
	auto v_tmp = v; //copie
	
	auto start = std::chrono::system_clock::now();
	
		#pragma omp parallel for
		for(std::size_t i=1;i<N-1; ++i)
		{
			for(std::size_t j=1;j<N-1; ++j)
			{

				for(std::size_t k=0; k<N-8; k+=8)
				{
					//celui en haut
					auto l0 = _mm256_loadu_ps(&v[i*N2+(j-1)*N + k+1]);
					
					//Celui du centre
					auto l1_1 = _mm256_loadu_ps(&v[i*N2+j*N+k]);
					auto l1_2 = _mm256_loadu_ps(&v[i*N2+j*N+k+1]);
					auto l1_3 = _mm256_loadu_ps(&v[i*N2+j*N+k+2]);
					
					//celui en dessous
					auto l2 = _mm256_loadu_ps(&v[i*N2+(j+1)*N + k+1]);
									
					//derriere
					auto lz0 = _mm256_loadu_ps(&v[(i-1)*N2+j*N + k+1]);
					//devant
					auto lz1 = _mm256_loadu_ps(&v[(i+1)*N2+j*N + k+1]);
					
					//calcul
					auto tmp1 = _mm256_mul_ps(l0,l0);
					auto soluce = _mm256_add_ps(tmp1, _mm256_mul_ps(l1_1,l1_1));
					soluce = _mm256_add_ps(soluce, _mm256_mul_ps(l1_2,l1_2));
					soluce = _mm256_add_ps(soluce, _mm256_mul_ps(l1_3,l1_3));
					soluce = _mm256_add_ps(soluce, _mm256_mul_ps(l2,l2));
					soluce = _mm256_add_ps(soluce, _mm256_mul_ps(lz0,lz0)); //soluce += lz0*lz0
					soluce = _mm256_add_ps(soluce, _mm256_mul_ps(lz1,lz1));
				
					//store dans le temporaire
					_mm256_storeu_ps(&v_tmp[i*N2+j*N+k+1],soluce);
				
				}
				//Pour les 6 derniers valeurs de la ligne
				for(std::size_t k=N-8;k<N-1;k++)
				{
					float x1,x2,y1,y2,z1,z2;
					x1=v[(i+1)*N2+j*N+k];
					x2=v[(i-1)*N2+j*N+k];
					
					y1=v[i*N2+(j+1)*N+k];
					y2=v[i*N2+(j-1)*N+k];
					
					z1=v[i*N2+j*N+(k+1)];
					z2=v[i*N2+j*N+(k-1)];
					
					float tmp = x1*x1+x2*x2+y1*y1+y2*y2+z1*z1+z2*z2;
					
					v_tmp[i*N2+j*N+k] = (v[i*N2+j*N+k]*v[i*N2+j*N+k] + tmp);
				}
			}
			
		}
		
	v=v_tmp;
	auto stop = std::chrono::system_clock::now();
	std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(stop-start).count() << "ms"<< std::endl;
}

/**
 * temps 246ms | 1354 ms sans -O3
 * */
void version_mm256_avecLoadU_sansStoreU()
{
	vec v(N*N*N,1.0f);
	auto v_tmp = v; //copie
	
	auto start = std::chrono::system_clock::now();
	
		#pragma omp parallel for
		for(std::size_t i=1;i<N-1; ++i)
		{
			for(std::size_t j=1;j<N-1; ++j)
			{
				__m256 reste;
				for(std::size_t k=0; k<N-8; k+=8)
				{	
					//celui en haut
					auto l0 = _mm256_loadu_ps(&v[i*N2+(j-1)*N + k+1]);
					
					//Celui du centre
					auto l1_1 = _mm256_loadu_ps(&v[i*N2+j*N+k]);
					auto l1_2 = _mm256_loadu_ps(&v[i*N2+j*N+k+1]);
					auto l1_3 = _mm256_loadu_ps(&v[i*N2+j*N+k+2]);
					
					//celui en dessous
					auto l2 = _mm256_loadu_ps(&v[i*N2+(j+1)*N + k+1]);
									
					//derriere
					auto lz0 = _mm256_loadu_ps(&v[(i-1)*N2+j*N + k+1]);
					//devant
					auto lz1 = _mm256_loadu_ps(&v[(i+1)*N2+j*N + k+1]);
					
					//calcul
					auto tmp1 = _mm256_mul_ps(l0,l0);
					auto soluce = _mm256_add_ps(tmp1, _mm256_mul_ps(l1_1,l1_1));
					soluce = _mm256_add_ps(soluce, _mm256_mul_ps(l1_2,l1_2));
					soluce = _mm256_add_ps(soluce, _mm256_mul_ps(l1_3,l1_3));
					soluce = _mm256_add_ps(soluce, _mm256_mul_ps(l2,l2));
					soluce = _mm256_add_ps(soluce, _mm256_mul_ps(lz0,lz0)); //soluce += lz0*lz0
					soluce = _mm256_add_ps(soluce, _mm256_mul_ps(lz1,lz1));
				
					//store dans le temporaire
					if(k==0)
						reste = l1_1;
					reste = storeShift1LFor3D(v_tmp,reste,l1_2,soluce,i,j,k);
					
				}
				//Pour les 6 derniers valeurs de la ligne
				for(std::size_t k=N-8;k<N-1;k++)
				{
					float x1,x2,y1,y2,z1,z2;
					x1=v[(i+1)*N2+j*N+k];
					x2=v[(i-1)*N2+j*N+k];
					
					y1=v[i*N2+(j+1)*N+k];
					y2=v[i*N2+(j-1)*N+k];
					
					z1=v[i*N2+j*N+(k+1)];
					z2=v[i*N2+j*N+(k-1)];
					
					float tmp = x1*x1+x2*x2+y1*y1+y2*y2+z1*z1+z2*z2;
					
					v_tmp[i*N2+j*N+k] = (v[i*N2+j*N+k]*v[i*N2+j*N+k] + tmp);
				}
			}
			
		}
	v=v_tmp;
	auto stop = std::chrono::system_clock::now();
	std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(stop-start).count() << "ms"<< std::endl;
}

/**
 * Version 234 ms | sans -03 1859ms
 * 
 * */
void version_mm256_sansLoadU_avecStoreU()
{
	vec v(N*N*N,1.0f);

	auto v_tmp = v; //copie
	
	auto start = std::chrono::system_clock::now();
	
		#pragma omp parallel for
		for(std::size_t i=1;i<N-1; ++i)
		{	
			for(std::size_t j=1;j<N-1; ++j)
			{
				//Pour chaque nouvelle ligne on charge 6
			
				//Celui du centre
				auto l1_1 = _mm256_load_ps(&v[i*N2+j*N]);
				auto l1_2 = _mm256_load_ps(&v[i*N2+j*N+8]);
				auto l1_res_shift2 = shift2(l1_1,l1_2);
				auto l1_res_shift1 = shift1(l1_1,l1_2);
				
				//celui au dessus
				auto l0_1 = _mm256_load_ps(&v[i*N2+(j-1)*N]);
				auto l0_2 = _mm256_load_ps(&v[i*N2+(j-1)*N+8]);
				auto l0_res = shift1(l0_1,l0_2);
				
				//celui en dessous
				auto l2_1 = _mm256_load_ps(&v[i*N2+(j+1)*N]);
				auto l2_2 = _mm256_load_ps(&v[i*N2+(j+1)*N+8]);
				auto l2_res = shift1(l2_1,l2_2);
				
				//celui derrière
				auto lz_1 = _mm256_load_ps(&v[(i-1)*N2+j*N]);
				auto lz_2 = _mm256_load_ps(&v[(i-1)*N2+j*N+8]);
				auto lz_res = shift1(lz_1,lz_2);
				
				//celui devant
				auto lz1_1 = _mm256_load_ps(&v[(i+1)*N2+j*N]);
				auto lz1_2 = _mm256_load_ps(&v[(i+1)*N2+j*N+8]);
				auto lz1_res = shift1(lz1_1,lz1_2);
				
				//calcul
				/**Partie avec FMA **/
				
				auto tmp1 = _mm256_mul_ps(l0_res,l0_res);
				auto soluce = _mm256_add_ps(tmp1, _mm256_mul_ps(l1_1,l1_1));
				soluce = _mm256_add_ps(soluce, _mm256_mul_ps(l1_res_shift2,l1_res_shift2));
				soluce = _mm256_add_ps(soluce, _mm256_mul_ps(l1_res_shift1,l1_res_shift1));
				soluce = _mm256_add_ps(soluce, _mm256_mul_ps(l2_res,l2_res));
				soluce = _mm256_add_ps(soluce, _mm256_mul_ps(lz_res,lz_res));
				soluce = _mm256_add_ps(soluce, _mm256_mul_ps(lz1_res,lz1_res));
				
				_mm256_storeu_ps(&v_tmp[i*N2+j*N+1],soluce);
				
				//On n'a plus qu'a continuer
				for(std::size_t k=8;k<N-8; k+=8)
				{
					//on copie les précédents
					l1_1 = l1_2;
					l0_1 = l0_2;
					l2_1 = l2_2;
					lz_1 = lz_2;
					lz1_1 = lz1_2;
					
					
					//On charge les 3 suivants
					l1_2 = _mm256_load_ps(&v[i*N2 + j*N +(k+8)]);
					
					l0_2 = _mm256_load_ps(&v[i*N2 + (j-1)*N +(k+8)]);
					l2_2 = _mm256_load_ps(&v[i*N2 + (j+1)*N +(k+8)]);
					
					lz_2 = _mm256_load_ps(&v[(i-1)*N2 + j*N +(k+8)]);
					lz1_2 = _mm256_load_ps(&v[(i+1)*N2 + j*N +(k+8)]);
					
					//les shifts nécessaires
					l1_res_shift2 = shift2(l1_1,l1_2);
					l1_res_shift1 = shift1(l1_1,l1_2);
					
					l0_res = shift1(l0_1,l0_2);
					
					l2_res = shift1(l2_1,l2_2);
					
					lz_res = shift1(lz_1,lz_2);
					
					lz1_res = shift1(lz1_1,lz1_2);
					
					//calcul
					
					/**Partie avec FMA **/
					
					tmp1 = _mm256_mul_ps(l0_res,l0_res);
					soluce = _mm256_add_ps(tmp1, _mm256_mul_ps(l1_1,l1_1));
					soluce = _mm256_add_ps(soluce, _mm256_mul_ps(l1_res_shift2,l1_res_shift2));
					soluce = _mm256_add_ps(soluce, _mm256_mul_ps(l1_res_shift1,l1_res_shift1));
					soluce = _mm256_add_ps(soluce, _mm256_mul_ps(l2_res,l2_res));
					soluce = _mm256_add_ps(soluce, _mm256_mul_ps(lz_res,lz_res));
					soluce = _mm256_add_ps(soluce, _mm256_mul_ps(lz1_res,lz1_res));
					
					
					//store dans le temporaire
					_mm256_storeu_ps(&v_tmp[i*N2+j*N+k+1],soluce);
					
				}
				//Pour les 6 derniers valeurs de la ligne
				for(std::size_t k=N-8;k<N-1;k++)
				{
					float x1,x2,y1,y2,z1,z2;
					x1=v[(i+1)*N2+j*N+k];
					x2=v[(i-1)*N2+j*N+k];
					
					y1=v[i*N2+(j+1)*N+k];
					y2=v[i*N2+(j-1)*N+k];
					
					z1=v[i*N2+j*N+(k+1)];
					z2=v[i*N2+j*N+(k-1)];
					
					float tmp = x1*x1+x2*x2+y1*y1+y2*y2+z1*z1+z2*z2;
					
					v_tmp[i*N2+j*N+k] = (v[i*N2+j*N+k]*v[i*N2+j*N+k]  + tmp);
				}
				
			}
			
			
			
		}

	v=v_tmp;
	
	auto stop = std::chrono::system_clock::now();
	std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(stop-start).count() << "ms"<< std::endl;
	
	
}

/**
 * Version 234 ms | sans -03 1859ms
 * 
 * */
void version_mm256_sansLoadU_sansStoreU()
{
	vec v(N*N*N,1.0f);

	auto v_tmp = v; //copie
	
	auto start = std::chrono::system_clock::now();
	
		#pragma omp parallel for
		for(std::size_t i=1;i<N-1; ++i)
		{	
			for(std::size_t j=1;j<N-1; ++j)
			{
				//Pour chaque nouvelle ligne on charge 6
			
				//Celui du centre
				auto l1_1 = _mm256_load_ps(&v[i*N2+j*N]);
				auto l1_2 = _mm256_load_ps(&v[i*N2+j*N+8]);
				auto l1_res_shift2 = shift2(l1_1,l1_2);
				auto l1_res_shift1 = shift1(l1_1,l1_2);
				
				//celui au dessus
				auto l0_1 = _mm256_load_ps(&v[i*N2+(j-1)*N]);
				auto l0_2 = _mm256_load_ps(&v[i*N2+(j-1)*N+8]);
				auto l0_res = shift1(l0_1,l0_2);
				
				//celui en dessous
				auto l2_1 = _mm256_load_ps(&v[i*N2+(j+1)*N]);
				auto l2_2 = _mm256_load_ps(&v[i*N2+(j+1)*N+8]);
				auto l2_res = shift1(l2_1,l2_2);
				
				//celui derrière
				auto lz_1 = _mm256_load_ps(&v[(i-1)*N2+j*N]);
				auto lz_2 = _mm256_load_ps(&v[(i-1)*N2+j*N+8]);
				auto lz_res = shift1(lz_1,lz_2);
				
				//celui devant
				auto lz1_1 = _mm256_load_ps(&v[(i+1)*N2+j*N]);
				auto lz1_2 = _mm256_load_ps(&v[(i+1)*N2+j*N+8]);
				auto lz1_res = shift1(lz1_1,lz1_2);
				
				//calcul
				/**Partie avec FMA **/
				//6 FMA
				auto tmp1 = _mm256_mul_ps(l0_res,l0_res);
				auto soluce = _mm256_add_ps(tmp1, _mm256_mul_ps(l1_1,l1_1));
				soluce = _mm256_add_ps(soluce, _mm256_mul_ps(l1_res_shift2,l1_res_shift2));
				soluce = _mm256_add_ps(soluce, _mm256_mul_ps(l1_res_shift1,l1_res_shift1));
				soluce = _mm256_add_ps(soluce, _mm256_mul_ps(l2_res,l2_res));
				soluce = _mm256_add_ps(soluce, _mm256_mul_ps(lz_res,lz_res));
				soluce = _mm256_add_ps(soluce, _mm256_mul_ps(lz1_res,lz1_res));
				
				
				//store dans le temporaire
				auto reste = storeShift1LFor3D(v_tmp,l1_1,l1_2,soluce,i,j,0);
				
				//On n'a plus qu'a continuer
				for(std::size_t k=8;k<N-8; k+=8)
				{
					//on copie les précédents
					l1_1 = l1_2;
					l0_1 = l0_2;
					l2_1 = l2_2;
					lz_1 = lz_2;
					lz1_1 = lz1_2;
					
					
					//On charge les 3 suivants
					l1_2 = _mm256_load_ps(&v[i*N2 + j*N +(k+8)]);
					
					l0_2 = _mm256_load_ps(&v[i*N2 + (j-1)*N +(k+8)]);
					l2_2 = _mm256_load_ps(&v[i*N2 + (j+1)*N +(k+8)]);
					
					lz_2 = _mm256_load_ps(&v[(i-1)*N2 + j*N +(k+8)]);
					lz1_2 = _mm256_load_ps(&v[(i+1)*N2 + j*N +(k+8)]);
					
					//les shifts nécessaires
					l1_res_shift2 = shift2(l1_1,l1_2);
					l1_res_shift1 = shift1(l1_1,l1_2);
					
					l0_res = shift1(l0_1,l0_2);
					
					l2_res = shift1(l2_1,l2_2);
					
					lz_res = shift1(lz_1,lz_2);
					
					lz1_res = shift1(lz1_1,lz1_2);
					
					//calcul
					
					/**Partie avec FMA **/
					
					tmp1 = _mm256_mul_ps(l0_res,l0_res);
					soluce = _mm256_add_ps(tmp1, _mm256_mul_ps(l1_1,l1_1));
					soluce = _mm256_add_ps(soluce, _mm256_mul_ps(l1_res_shift2,l1_res_shift2));
					soluce = _mm256_add_ps(soluce, _mm256_mul_ps(l1_res_shift1,l1_res_shift1));
					soluce = _mm256_add_ps(soluce, _mm256_mul_ps(l2_res,l2_res));
					soluce = _mm256_add_ps(soluce, _mm256_mul_ps(lz_res,lz_res));
					soluce = _mm256_add_ps(soluce, _mm256_mul_ps(lz1_res,lz1_res));
					
					
					/**Partie avec juste ADD pour vérifier**/
					/*soluce = _mm256_add_ps(l0_res, l1_1);
					soluce = _mm256_add_ps(soluce, l1_res_shift2);
					soluce = _mm256_add_ps(soluce, l1_res_shift1);
					soluce = _mm256_add_ps(soluce, l2_res);*/
					
					
					//store dans le temporaire					
					reste = storeShift1LFor3D(v_tmp,reste,l1_2,soluce,i,j,k);
				}
				//Pour les 6 derniers valeurs de la ligne
				for(std::size_t k=N-8;k<N-1;k++)
				{
					float x1,x2,y1,y2,z1,z2;
					x1=v[(i+1)*N2+j*N+k];
					x2=v[(i-1)*N2+j*N+k];
					
					y1=v[i*N2+(j+1)*N+k];
					y2=v[i*N2+(j-1)*N+k];
					
					z1=v[i*N2+j*N+(k+1)];
					z2=v[i*N2+j*N+(k-1)];
					
					float tmp = x1*x1+x2*x2+y1*y1+y2*y2+z1*z1+z2*z2;
					
					v_tmp[i*N2+j*N+k] = (v[i*N2+j*N+k]*v[i*N2+j*N+k]  + tmp);
				}
				
			}
			
			
			
		}
		
	v=v_tmp;
	
	auto stop = std::chrono::system_clock::now();
	std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(stop-start).count() << "ms"<< std::endl;
	
	
}
int main()
{	
	std::cout << "version classic:\n";
	version_scalaire();
	
	std::cout << "version avec les _mm256 sans loadu sans storeu\n";
	version_mm256_sansLoadU_sansStoreU();
	
	std::cout << "version avec les _mm256 avec loadu sans storeu\n";
	version_mm256_avecLoadU_sansStoreU();
	
	
	std::cout << "version avec les _mm256 sans loadu avec storeu\n";
	version_mm256_avecLoadU_sansStoreU();
	
	std::cout << "version avec les _mm256 avec loadu avec storeu\n";
	version_mm256_avecLoadU_avecStoreU();
	return 0;
}
