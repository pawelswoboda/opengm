#include "mex_io.h"

template<typename type> mxClassID mexClassId(){
	throw debug_exception(String("Type is not recognized: ")+typeid(type).name());
};

template<> mxClassID mexClassId<char>(){
	return mxINT8_CLASS;
};

template<> mxClassID mexClassId<unsigned char>(){
	return mxUINT8_CLASS;
};

template<> mxClassID mexClassId<short>(){
	return mxINT16_CLASS;
};

template<> mxClassID mexClassId<unsigned short>(){
	return mxUINT16_CLASS;
};

template<> mxClassID mexClassId<int>(){
	return mxINT32_CLASS;
};

template<> mxClassID mexClassId<unsigned int>(){
	return mxUINT32_CLASS;
};

template<> mxClassID mexClassId<long long>(){
	return mxINT64_CLASS;
};

template<> mxClassID mexClassId<unsigned long long>(){
	return mxUINT64_CLASS;
};

template<> mxClassID mexClassId<float>(){
	return mxSINGLE_CLASS;
};

template<> mxClassID mexClassId<double>(){
	return mxDOUBLE_CLASS;
};

template<> mxClassID mexClassId<bool>(){
	return mxLOGICAL_CLASS;
};

template<> mxClassID mexClassId<char*>(){
	return mxCHAR_CLASS;
};

template<> mxClassID mexClassId<mx_struct>(){
	return mxSTRUCT_CLASS;
};

Engine *matlab = 0;

void start_engine(){
	std::cout << "Starting MATLAB engine" << std::endl;
	if (!(matlab = engOpen(NULL))){
		throw debug_exception("Cant start matlab Engine");
	};
	engSetVisible(matlab, true);
};
