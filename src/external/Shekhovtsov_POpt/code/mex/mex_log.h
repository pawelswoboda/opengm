#ifndef mex_log_h 
#define mex_log_h 

#include <mex.h>

#include "streams/xstringstream.h"
#include "mex_io.h"

namespace mexargs{
	using namespace exttype;
	using txt::String;
	class MexStream : public txt::TextStream{
	private:
		virtual TextStream & write(const char * x){
			mexPrintf("%s",x);
			return *this;
		};
	public:
	};

	class MexLogStream: public txt::pTextStream{
	public:
		MexLogStream(std::string filename, bool append=true){
			attach(new txt::TabbedTextStream(txt::EchoStream(txt::FileStream(filename.c_str(),append),MexStream())),true);
		};
		~MexLogStream(){
			detach();
		};
		//		static MexLogStream log;
	};
};

#endif