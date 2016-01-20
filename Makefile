XGB_INCLUDE_DIRS = ${XGB_REPO}/wrapper \
				   ${XGB_REPO}/src/io

IXGB_INCLUDE_DIRS = $(patsubst %,-I%,${XGB_INCLUDE_DIRS})

build/tag: models/ele_trained.xgb data/nnet_ele.xmat build/tag.o
	${CXX} -fopenmp -std=c++11 build/tag.o -L${XGB_REPO}/wrapper -lxgboostwrapper -o $@

build/tag.o: tag.cxx | build
	${CXX} -fopenmp -std=c++11 -c tag.cxx -o $@ ${IXGB_INCLUDE_DIRS}

build:
	mkdir -p build

clean:
	rm -rf build

cleanall:
	rm -rf build models
