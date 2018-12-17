rm -rf build

mkdir -p build
cd build
cmake -DCMAKE_PREFIX_PATH=../../../libtorch ..
make -j4
cd ..

mv build/predict .

rm -rf build
