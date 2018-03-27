CC=g++ -std=c++11

CCFLAGS =-march=native 

DIR_BOOST =-I /home/etud/o2121076/Modèles/boost_1_66_0


all: stencil3d stencil3d_O3 stencil2d stencil2d_O3

stencil2d_O3: stencil.cpp
	$(CC) $^ -o $@ -O3 $(CCFLAGS) $(DIR_BOOST)
	
stencil2d: stencil.cpp
	$(CC) $^ -o $@ $(CCFLAGS) $(DIR_BOOST)
	
stencil3d_O3: stencil3D.cpp
	$(CC) $^ -o $@ -O3 $(CCFLAGS) $(DIR_BOOST)
	
stencil3d: stencil3D.cpp
	$(CC) $^ -o $@ $(CCFLAGS) $(DIR_BOOST)

clean:
	rm -f stencil2d_O3
	rm -f stencil2d
	rm -f stencil3d_O3
	rm -f stencil3d
