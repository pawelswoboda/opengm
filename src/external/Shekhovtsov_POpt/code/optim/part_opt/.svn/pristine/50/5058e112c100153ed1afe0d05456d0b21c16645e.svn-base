//UnrollerP: loops over given size, partial unrolled
template<int InnerUnroll = 8, int Begin = 0> class Unroller{
public:
	struct UnrollerP {
		template<typename Lambda>
		static void __forceinline step(int N, Lambda&& func) {
			int i = Begin;
			for (; i < N; ++i){ // do the reminded by simple loop
				UnrollerInternal<1>::step(func, i);
			};
			
			/*
			for (; i < N - InnerUnroll + 1; i += InnerUnroll) {
				UnrollerInternal<InnerUnroll>::step(func, i);
			}
			
			for (; i < N; ++i){ // do the reminded by simple loop
				UnrollerInternal<1>::step(func, i);
			};
			
			switch (N - i){
			case 0:return;
			case 1:return UnrollerInternal<1>::step(func, i);
			case 2:return UnrollerInternal<2>::step(func, i);
			case 3:return UnrollerInternal<3>::step(func, i);
			case 4:return UnrollerInternal<4>::step(func, i);
			case 5:return UnrollerInternal<5>::step(func, i);
			case 6:return UnrollerInternal<6>::step(func, i);
			case 7:return UnrollerInternal<7>::step(func, i);
			default: assert(false);
			};
			*/
			// reminder is at most InnerUnroll-1, unroll by if
			//UnrollerReminder<InnerUnroll - 1>::step(i, N, func);
		}
	private:
		template<int Reminder, class DUMMY = void>
		struct UnrollerReminder {
			template<typename Lambda>
			static void __forceinline step(int i, int N, Lambda&& func) {
				if (N - i == Reminder){
					UnrollerInternal<Reminder>::step(func, i);
				} else{
					UnrollerReminder<Reminder - 1>::step(i, N, func);
				};
			}
		};
		template<class DUMMY>
		struct UnrollerReminder < 0, DUMMY> {
			template<typename Lambda>
			static void __forceinline step(int i, int N, Lambda&& func) {
				//nothing to do
			}
		};
		//start of UnrollerInternal
		template<int Unroll, int Offset = 0>
		struct UnrollerInternal {
			template<typename Lambda>
			static void __forceinline step(Lambda&& func, int i) {
				func(i + Offset);
				UnrollerInternal<Unroll, Offset + 1>::step(func, i);
			}
		};
		//end of UnrollerInternal
		template<int Unroll>
		struct UnrollerInternal < Unroll, Unroll > {
			template<typename Lambda>
			static void __forceinline step(Lambda&& func, int i) {
			}
		};

	};
};