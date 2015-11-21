#ifndef GEMUCUDATHREADSH
#define GEMUCUDATHREADSH

#include <vector>
#include <iostream>
#include <memory>

namespace gemu {
	namespace cuda {

		template <typename T>
		struct pt3d {
			T x, y, z;
			pt3d(T a=T(), T b=T(), T c=T()) : x(a), y(b), z(c) {}
			bool operator==(const pt3d& other) const {
				return x==other.x && y==other.y && z==other.z;
			}
		};
		typedef pt3d<size_t> dim3;

		template <typename T>
		class Array3D {
		public:
			Array3D(const dim3& size, T elemTemplate)
			:_size(size)
			,_data(std::vector< std::vector<std::vector<T>> >(size.x,
							std::vector<std::vector<T>>(size.y,
								std::vector<T>(size.z, elemTemplate))))
			{
				for (size_t i=0 ; i<size.x ; ++i) {
					for (size_t j=0 ; j<size.y ; ++j) {
						for (size_t k=0 ; k<size.z ; ++k) {
							_data[i][j][k].setPos(i,j,k);
						}
					}
				}
			}
			const size_t count() const {
				return this->_size.x * this->_size.y * this->_size.z;
			}
			const dim3& size() const {
				return this->_size;
			}
			T& get(size_t x, size_t y, size_t z=0) {
				return this->_data[x][y][z];
			}
			T& get(const dim3& pos){
				return this->get(pos.x, pos.y, pos.z);
			}
			const T get(size_t x, size_t y, size_t z=0) const {
				return this->_data[x][y][z];
			}
			const T get(const dim3& pos) const{
				return this->get(pos.x, pos.y, pos.z);
			}
		private:
			const dim3 _size;
			std::vector< std::vector< std::vector<T> > > _data;
		};

		class ThreadBlock;
		class ThreadGrid;

		class Thread {
		public:
			Thread(){}
			Thread(ThreadBlock& block){}
			Thread(const Thread& other)
			:_pos(other._pos)
			{}
			Thread& operator=(const Thread& other){
				this->_pos = other._pos;
				return *this;
			}
			void setPos(size_t x, size_t y, size_t z=0) {
				this->_pos = dim3(x,y,z);
			}
			const dim3& pos() const {
				return this->_pos;
			}
		private:
			dim3 _pos;
			friend class ThreadBlock;
		};

		class ThreadBlock {
		public:
			ThreadBlock(const dim3& size, ThreadGrid& grid)
			:_grid(grid)
			{
				_threads = new Array3D<Thread>(size, Thread(*this));
			}
			ThreadBlock(ThreadBlock&& other)
			:_pos(std::move(other._pos))
			,_threads(std::move(other._threads))
			,_grid(other._grid)
			{
				other._threads = nullptr;
			}
			~ThreadBlock(){
				delete this->_threads;
			}
			Thread thread(const dim3& pos) const {
				return this->_threads->get(pos);
			}
			Thread thread(size_t x, size_t y, size_t z) const {
				return this->_threads->get(x, y, z);
			}
			const dim3& size() const {
				return this->_threads->size();
			}
			const size_t threadCount() const {
				return this->_threads->count();
			}
			const dim3& pos() const {
				return this->_pos;
			}
			void setPos(size_t x, size_t y, size_t z=0) {
				this->_pos = dim3(x,y,z);
			}
			ThreadGrid& grid(){return this->_grid;}
		private:
			dim3 _pos;
			Array3D<Thread> * _threads;
			ThreadGrid& _grid;
			friend class ThreadGrid;
		private:
			ThreadBlock(const ThreadBlock&);
		};
		typedef std::shared_ptr<ThreadBlock> ThreadBlockPtr;
		class ThreadGrid {
		public:
            ThreadGrid(const dim3& gridSize=dim3(), const dim3& blockSize=dim3())
				:_size(gridSize)
				,_blockSize(blockSize)
			{

			}
			~ThreadGrid(){

			}
			ThreadBlockPtr block(size_t x, size_t y, size_t z) {
				if (x >= this->_size.x || y >= this->_size.y || z >= this->_size.z)
					return ThreadBlockPtr(nullptr);
				ThreadBlockPtr result(new ThreadBlock(this->_blockSize, *this));
				result->setPos(x, y, z);
				return result;
			}
			ThreadBlockPtr block(const dim3& pos) {
				return this->block(pos.x, pos.y, pos.z);
			}
			ThreadBlockPtr block(size_t i) {
				const size_t nxy = this->_size.x * this->_size.y;
				const size_t nz = i / nxy;
				const size_t nx = i % this->_size.x;
				const size_t ny = (i - nz * nxy) / this->_size.x;
				return this->block(nx, ny, nz);
			}
			const size_t blockCount() const {
				return this->_size.x * this->_size.y * this->_size.z;
			}
			const dim3& size() const {
				return this->_size;
			}
		private:
            dim3 _size;
            dim3 _blockSize;
		};
	}
}

#endif
