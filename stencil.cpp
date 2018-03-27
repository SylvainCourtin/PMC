#include <vector>
#include <chrono>
#include <iostream>

#include <boost/align/aligned_allocator.hpp>
#include <immintrin.h>

//Pour le alignas : __attribute__((aligned(32)))
// float alignas(32) v0[8]={1,2,3,4,5,6,7,8};
//alignas(32) float v0[8]={1,2,3,4,5,6,7,8};

using vec = std::vector<float,boost::alignment::aligned_allocator<float,32> >;

static std::size_t const N=512;
static int val=1;

void affiche2D(vec v)
{ 
	for(std::size_t i=0;i<N; i++)
	{
		for(std::size_t j=0;j<N; j++)
		{
			std::cout << v[i*N+j] << ",";
		}
		std::cout << "\n";
	}
}

//Version naïf
/**
 * Temps ~120ms avec -O3 | ~4013ms sans -O3
 * On pourra augmenter N pour augmenter le temps afin de mieux comparer les temps d'execution
 * */
void version_scalaire()
{	
	std::vector<float> v(N*N,1.0f);
	std::vector<float> v_tmp(N*N);
	v_tmp = v;
	
	auto start = std::chrono::system_clock::now();
	
	for(int x=0;x<1000;x++)
	{
		#pragma omp parallel for
		for(std::size_t i=1;i<N-1; i++)
		{
			for(std::size_t j=1;j<N-1; j++)
			{
				
				float h(0),b(0),g(0),d(0);
				
				h=v[(i)* N+j-1];
				b=v[(i)* N+j+1];
				g=v[(i-1)* N+j];
				d=v[(i+1)* N+j];
				
				float tmp = h*h+b*b+g*g+d*d;
				
				v_tmp[i*N+j] = (v[i*N+j]* v[i*N+j] + tmp);
			}
			
		}
	}
	v = v_tmp;
	
	auto stop = std::chrono::system_clock::now();
	std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(stop-start).count() << "ms"<< std::endl;	
	
}

/**
 * Pour la partie shift
 * 
 * */

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

/*******************************************************************************/

/**
 * Pour la partie store
 * 
 * */
 
 inline __m256 storeShift1LFor(vec &v_tmp, __m256 a, __m256 b, __m256 soluce, size_t i, size_t j){
	
	auto permuteG = _mm256_permute_ps(soluce,0x93);
	auto permute2f128 = _mm256_permute2f128_ps(permuteG,permuteG,0x01);
	auto blend1 = _mm256_blend_ps(permuteG, a, 0x01);
	auto blend1_2 = _mm256_blend_ps (blend1, permute2f128, 0x10);
	
	auto blend2 = _mm256_blend_ps (b, permute2f128, 0x01);
	
	_mm256_store_ps(&v_tmp[i*N+j],blend1_2);
	_mm256_store_ps(&v_tmp[i*N+(j+8)],blend2);
	
	return blend2;
}
/*******************************************************************************/

/**
 * temps avec -O3 ~110ms | sans -O3 
 * */
void version_mm256_avecLoadU()
{
	vec v(N*N,1.0f);
	auto v_tmp = v; //copie
	
	auto start = std::chrono::system_clock::now();
	
	for(int x=0;x<1000;x++)
	{
		#pragma omp parallel for
		for(std::size_t i=1;i<N-1; ++i)
		{
			for(std::size_t j=0;j<N-8; j+=8)
			{
				//celui au dessus
				auto l0 = _mm256_loadu_ps(&v[(i-1)*N+1+j]);
				
				//Celui du centre
				auto l1_1 = _mm256_loadu_ps(&v[i*N+j]);
				auto l1_2 = _mm256_loadu_ps(&v[i*N+1+j]);
				auto l1_3 = _mm256_loadu_ps(&v[i*N+2+j]);
				
				//celui en dessous
				auto l2 = _mm256_loadu_ps(&v[(i+1)*N+1+j]);
				
				//calcul
				auto tmp1 = _mm256_mul_ps(l0,l0);
				auto soluce = _mm256_add_ps(tmp1, _mm256_mul_ps(l1_1,l1_1));
				soluce = _mm256_add_ps(soluce, _mm256_mul_ps(l1_2,l1_2));
				soluce = _mm256_add_ps(soluce, _mm256_mul_ps(l1_3,l1_3));
				soluce = _mm256_add_ps(soluce, _mm256_mul_ps(l2,l2));
				
				//store dans le temporaire
				_mm256_storeu_ps(&v_tmp[i*N+(j+1)],soluce);
				
				
			}
			//Pour les 6 derniers valeurs de la ligne
			for(std::size_t j=N-8;j<N-1;j++)
			{
				float h(0),b(0),g(0),d(0);
				
				h=v[(i)* N+j-1];
				b=v[(i)* N+j+1];
				g=v[(i-1)* N+j];
				d=v[(i+1)* N+j];
				
				float tmp = h*h+b*b+g*g+d*d;
				
				v_tmp[i*N+j] = (v[i*N+j]*v[i*N+j] + tmp) ;
			}
		}
	}
	v=v_tmp;
	auto stop = std::chrono::system_clock::now();
	std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(stop-start).count() << "ms"<< std::endl;
	
	//affiche2D(v);
}

/**
 * Version avec -O3 ~100ms | sans -O3 ~1200ms
 * 
 * */
void version_mm256_sansLoadU()
{
	vec v(N*N,1.0f);
	auto v_tmp = v; //copie
	auto start = std::chrono::system_clock::now();
	
	for(int x=0;x<1000;x++){
		
		#pragma omp parallel for
		for(std::size_t i=1;i<N-1; ++i)
		{	//Pour chaque nouvelle ligne on charge 6
			
			//Celui du centre
			auto l1_1 = _mm256_load_ps(&v[i*N]);
			auto l1_2 = _mm256_load_ps(&v[i*N+8]);
			auto l1_res_shift2 = shift2(l1_1,l1_2);
			auto l1_res_shift1 = shift1(l1_1,l1_2);
			
			//celui au dessus
			auto l0_1 = _mm256_load_ps(&v[(i-1)*N]);
			auto l0_2 = _mm256_load_ps(&v[(i-1)*N+8]);
			auto l0_res = shift1(l0_1,l0_2);
			
			//celui en dessous
			auto l2_1 = _mm256_load_ps(&v[(i+1)*N]);
			auto l2_2 = _mm256_load_ps(&v[(i+1)*N+8]);
			auto l2_res = shift1(l2_1,l2_2);
			
			//calcul
			/**Partie avec juste ADD pour vérifier**/
			/*auto soluce = _mm256_add_ps(l0_res, l1_1);
			soluce = _mm256_add_ps(soluce, l1_res_shift2);
			soluce = _mm256_add_ps(soluce, l1_res_shift1);
			soluce = _mm256_add_ps(soluce, l2_res);*/
			/**Partie avec FMA **/
			
			auto tmp1 = _mm256_mul_ps(l0_res,l0_res);
			auto soluce = _mm256_add_ps(tmp1, _mm256_mul_ps(l1_1,l1_1));
			soluce = _mm256_add_ps(soluce, _mm256_mul_ps(l1_res_shift2,l1_res_shift2));
			soluce = _mm256_add_ps(soluce, _mm256_mul_ps(l1_res_shift1,l1_res_shift1));
			soluce = _mm256_add_ps(soluce, _mm256_mul_ps(l2_res,l2_res));
			
			
			//Recupere le 8eme bit de la soluce pour ne pas le perdre
			//auto rst = storeShift1LFor(v_tmp,l1_1,l1_2,soluce,i,0);
			_mm256_storeu_ps(&v_tmp[i*N+1],soluce);
			
			
			//On n'a plus qu'a continuer
			for(std::size_t j=8;j<N-8; j+=8)
			{
				//on copie les précédents
				l1_1 = l1_2;
				l0_1 = l0_2;
				l2_1 = l2_2;
				//On charge les 3 suivants
				l1_2 = _mm256_load_ps(&v[i*N + (j+8)]);
				l0_2 = _mm256_load_ps(&v[(i-1)*N + (j+8)]);
				l2_2 = _mm256_load_ps(&v[(i+1)*N + (j+8)]);
				
				//les shifts nécessaires
				l1_res_shift2 = shift2(l1_1,l1_2);
				l1_res_shift1 = shift1(l1_1,l1_2);
				
				l0_res = shift1(l0_1,l0_2);
				
				l2_res = shift1(l2_1,l2_2);
				
				//calcul
				
				/**Partie avec FMA **/
				
				tmp1 = _mm256_mul_ps(l0_res,l0_res);
				soluce = _mm256_add_ps(tmp1, _mm256_mul_ps(l1_1,l1_1));
				soluce = _mm256_add_ps(soluce, _mm256_mul_ps(l1_res_shift2,l1_res_shift2));
				soluce = _mm256_add_ps(soluce, _mm256_mul_ps(l1_res_shift1,l1_res_shift1));
				soluce = _mm256_add_ps(soluce, _mm256_mul_ps(l2_res,l2_res));
				
				
				/**Partie avec juste ADD pour vérifier**/
				/*soluce = _mm256_add_ps(l0_res, l1_1);
				soluce = _mm256_add_ps(soluce, l1_res_shift2);
				soluce = _mm256_add_ps(soluce, l1_res_shift1);
				soluce = _mm256_add_ps(soluce, l2_res);*/
				
				
				//store dans le temporaire
				_mm256_storeu_ps(&v_tmp[i*N+(j+1)],soluce);
				//rst = storeShift1LFor(v_tmp,rst,l1_2,soluce,i,j);
				
				
			}
			//Pour les 6 derniers valeurs de la ligne
			for(std::size_t j=N-8;j<N-1;j++)
			{
				float h(0),b(0),g(0),d(0);
				
				h=v[(i)* N+j-1];
				b=v[(i)* N+j+1];
				g=v[(i-1)* N+j];
				d=v[(i+1)* N+j];
				
				float tmp = h*h+b*b+g*g+d*d;
				
				v_tmp[i*N+j] = (v[i*N+j]*v[i*N+j] + tmp) ;
			}
		}
		
	}
	v=v_tmp;
	auto stop = std::chrono::system_clock::now();
	std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(stop-start).count() << "ms"<< std::endl;
	
	//affiche2D(v);
	
}

int main()
{	
	std::cout << "version classic:\n";
	version_scalaire();
	std::cout << "version avec les _mm256 sans loadu\n";
	version_mm256_sansLoadU();
	std::cout << "version avec les _mm256 avec loadu\n";
	version_mm256_avecLoadU();
	return 0;
}
