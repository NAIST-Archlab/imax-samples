#ifndef PROFILE_HPP
#define PROFILE_HPP

#if PROFILE

#include<map>
#include<string>
std::map< std::string , long double > time_sum;

#define START_TIME(name) \
	std::chrono::system_clock::time_point name ## _start; \
    name ## _start = std::chrono::system_clock::now()

#define END_TIME(name) \
    std::chrono::system_clock::time_point name ## _end; \
    name ## _end = std::chrono::system_clock::now(); \
    time_sum[#name] += static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(name ## _end - name ## _start).count() / 1000.0)

#define START_TIME2(name) \
    name ## _start = std::chrono::system_clock::now()

#define END_TIME2(name) \
    name ## _end = std::chrono::system_clock::now(); \
    time_sum[#name] += static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(name ## _end - name ## _start).count() / 1000.0)



#define PRINT_TIMES() \
	for( auto itr = time_sum.begin() ; itr !=  time_sum.end() ; itr++ ) printf("%s,\t%.5Lf\n" , itr->first.c_str() , itr->second );


#else

#define START_TIME(name)
#define END_TIME(name)
#define START_TIME2(name)
#define END_TIME2(name)
#define PRINT_TIMES()

#endif

#endif


