#include "performance.h"

#define TIME1

#if _MSC_VER > 1000
#ifdef TIME1
#include <windows.h>

namespace debug{
	PerformanceCounter::PerformanceCounter(){
		start();
		counting = true;
	};
	void PerformanceCounter::start(){
		QueryPerformanceCounter((LARGE_INTEGER*)&t1);
		Dt = 0;
		counting = true;
	};
	__int64 PerformanceCounter::tickcount()const{
		if(counting){
			__int64 t2;
			QueryPerformanceCounter((LARGE_INTEGER*)&t2);
			return (t2-t1+Dt);
		}else{
			return Dt;
		}
	};
	double PerformanceCounter::time()const{
		__int64 f;
		QueryPerformanceFrequency((LARGE_INTEGER*)&f);
		return double(tickcount())/f;
	};
	void PerformanceCounter::pause(){
		if(counting){
			__int64 t2;
			QueryPerformanceCounter((LARGE_INTEGER*)&t2);
			Dt = Dt+(t2-t1);
			t1 = t2;
			counting = false;
		};
	};

	void PerformanceCounter::stop(){
		pause();
	};

	void PerformanceCounter::resume(){
		if(!counting){
			QueryPerformanceCounter((LARGE_INTEGER*)&t1);
			counting = true;
		};
	};
};
#else
#include <time.h>

namespace debug{

	void QueryPerformanceCounter(long long * t){
		clock_t tt = clock();
		*t = tt;
	};

	void QueryPerformanceFrequency(long long *f){
		*f = CLOCKS_PER_SEC;
	};

	PerformanceCounter::PerformanceCounter(){
		start();
		counting = true;
	};
	void PerformanceCounter::start(){
		QueryPerformanceCounter(&t1);
		Dt = 0;
		counting = true;
	};
	long long PerformanceCounter::tickcount()const{
		if(counting){
			__int64 t2;
			QueryPerformanceCounter(&t2);
			return (t2-t1+Dt);
		}else{
			return Dt;
		}
	};
	double PerformanceCounter::time()const{
		__int64 f;
		QueryPerformanceFrequency(&f);
		return double(tickcount())/f;
	};
	void PerformanceCounter::pause(){
		__int64 t2;
		QueryPerformanceCounter(&t2);
		Dt = Dt+(t2-t1);
		t1 = t2;
		counting = false;
	};

	void PerformanceCounter::stop(){
		pause();
	};

	void PerformanceCounter::resume(){
		QueryPerformanceCounter(&t1);
		counting = true;
	};
};
#endif

#else // NOT MSVC
#ifdef __MACH__
	#include <sys/time.h>
	#include <stdio.h>
	#include <mach/clock.h>
	#include <mach/mach.h>
#else //NOT __MACH__
	#include <time.h>
#endif

namespace debug{

void current_utc_time(struct timespec *ts) {

#ifdef __MACH__ // OS X does not have clock_gettime, use clock_get_time
	clock_serv_t cclock;
	mach_timespec_t mts;
	host_get_clock_service(mach_host_self(), CALENDAR_CLOCK, &cclock);
	clock_get_time(cclock, &mts);
	mach_port_deallocate(mach_task_self(), cclock);
	ts->tv_sec = mts.tv_sec;
	ts->tv_nsec = mts.tv_nsec;
#else
	clock_gettime(CLOCK_REALTIME, ts);
#endif

}


	typedef long long __int64;

	const long long nano = 1000000000;

	void QueryPerformanceCounter(long long * t){
		timespec tt;
		//clock_gettime(CLOCK_REALTIME,&tt);
		current_utc_time(&tt);
		*t = tt.tv_nsec+((long long)(tt.tv_sec)*nano);
	};

	void QueryPerformanceFrequency(long long *f){
		*f = nano;
	};

	PerformanceCounter::PerformanceCounter(){
		start();
		counting = true;
	};
	void PerformanceCounter::start(){
		QueryPerformanceCounter(&t1);
		Dt = 0;
		counting = true;
	};
	long long PerformanceCounter::tickcount()const{
		if(counting){
			__int64 t2;
			QueryPerformanceCounter(&t2);
			return (t2-t1+Dt);
		}else{
			return Dt;
		}
	};
	double PerformanceCounter::time()const{
		__int64 f;
		QueryPerformanceFrequency(&f);
		return double(tickcount())/f;
	};
	void PerformanceCounter::pause(){
		__int64 t2;
		QueryPerformanceCounter(&t2);
		Dt = Dt+(t2-t1);
		t1 = t2;
		counting = false;
	};

	void PerformanceCounter::stop(){
		pause();
	};

	void PerformanceCounter::resume(){
		QueryPerformanceCounter(&t1);
		counting = true;
	};
};

#endif
