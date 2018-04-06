CC=g++ -std=c++11

CCFLAGS =-march=native 

BOOST_ROOT := /usr/local/boost_1_66_0
BOOST_INC := ${BOOST_ROOT}/include
#DIR_BOOST =-I /home/etud/thibault.machado/M2/PMC/boost_1_66_0 stencil.cpp


all: stencil3d stencil3d_O2 stencil3d_O3 stencil2d stencil2d_O2 stencil2d_O3

stencil2d_O3: stencil.cpp
	$(CC) $^ -o $@ -O3 $(CCFLAGS) -I$(BOOST_INC)

stencil2d_O2: stencil.cpp
	$(CC) $^ -o $@ -O2 $(CCFLAGS) -I$(BOOST_INC)
	
stencil2d: stencil.cpp
	$(CC) $^ -o $@ $(CCFLAGS) -I$(BOOST_INC)
	
stencil3d_O3: stencil3D.cpp
	$(CC) $^ -o $@ -O3 $(CCFLAGS) -I$(BOOST_INC)
		
stencil3d_O2: stencil3D.cpp
	$(CC) $^ -o $@ -O2 $(CCFLAGS) -I$(BOOST_INC)
	
stencil3d: stencil3D.cpp
	$(CC) $^ -o $@ $(CCFLAGS) -I$(BOOST_INC)

clean:
	rm -f stencil2d_O3
	rm -f stencil2d_O2
	rm -f stencil2d
	rm -f stencil3d_O3
	rm -f stencil3d_O2
	rm -f stencil3d
