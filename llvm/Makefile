all:
	mkdir build
	cd build; cmake ..; make
	mkdir bin

run:
	for file in src/*.c; do \
		OBJ=`echo "$file" | cut -d'.' -f1`; \
		clang -Xclang -load -Xclang build/skeleton/libSkeletonPass.* $$file -o $$OBJ; \
		mv $$OBJ bin; \
	done

clean:
	rm -rf bin
	rm -rf build

.SILENT:

