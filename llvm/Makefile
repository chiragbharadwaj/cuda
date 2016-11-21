all:
	mkdir build
	cd build; cmake ..; make
	mkdir bin

lib: all
	cc -c src/rtlib.c
	clang -Xclang -load -Xclang build/skeleton/libSkeletonPass.so -c src/add.c
	cc add.o rtlib.o -o add
	mv add add.o rtlib.o bin
	bin/add

pass: all
	for file in src/*.c; do \
		OBJ=`echo "$$file" | cut -d'.' -f1`; \
		clang -Xclang -load -Xclang build/skeleton/libSkeletonPass.* $$file -o $$OBJ; \
		mv $$OBJ bin; \
	done

clean:
	rm -rf bin
	rm -rf build

.SILENT:

