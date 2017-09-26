#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#ifndef M_PI
    #define M_PI 3.14159265358979323846
#endif

inline double4 get_accel(double4 pos, double4 v, float4* coils, int num_coils, float c_sphere, __global float* ee,__global float* ek) //where factor equals charge/mass
{

	float E_k = 0.0f;
	float K_k = 0.0f;

  double CR = .02; //charge radius

	float4 b = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
	float r = sqrt(pos.x*pos.x + pos.y*pos.y);
  double R_2 = pos.x * pos.x + pos.y*pos.y + (pos.z - coils[1].x/2.0)*(pos.z - coils[1].x/2.0);
	float B_r;

	float a;
	float B;
	float Q;
	int k;

    for( int i = 0; i < num_coils; i++ ){
		//z = (pos.z - coils[i].x);
		// z, radius, b0, rotation int

		a = r/coils[i].y;
		B = (pos.z - coils[i].x)/coils[i].y;

		Q = sqrt((1.0f + a) * (1.0f + a) + B*B);
		k = (int)(4.0f * a / (Q*Q) *10000000);

		E_k = ee[k];
		K_k = ek[k];


		B_r = coils[i].z * ((pos.z - coils[i].x)/r)/( M_PI * Q) * ( ( E_k * (1.0f + a*a + B*B) / ((Q*Q)-4.0f*a) ) - K_k);

		b.z += coils[i].z * (1.0f)/( M_PI * Q) * (( E_k * (1.0f - a*a - B*B)/((Q*Q)-4.0f*a)) + K_k);
		b.x += pos.x / r * B_r;
		b.y += pos.y / r * B_r;

	}

  double q = c_sphere * (fmin(sqrt(R_2)/(CR*CR*CR), 1.0/(CR*CR)) + fmin(1.0/R_2, 1.0/(CR*CR)) - 1.0/(CR*CR));
  double4 e_field = q * 8.99e9 * (double4)(sign(pos.x) * pos.x*pos.x/R_2, sign(pos.y) * pos.y*pos.y/R_2, sign(pos.z) * pos.z*pos.z/R_2, 0.0);
	return ((double4)( v.y * b.z - v.z * b.y, v.z * b.x - v.x * b.z, v.x * b.y - v.y * b.x, 0.0) + e_field) * -175882002272.0;
}



__kernel void compute_trajectory(
	__global double4* positions,			//xyz, charge/mass
	__global double4* velocities,		//xyz
	__global float4* coils,
  __global double* c_spheres,
	__global float* ee_tab,
	__global float* ek_tab,
	__global float4* dest,
  int4 sim_properties, // num_particles, num_steps, iter_nth, num_coils
	 double dt
	 ){
	unsigned int thread = get_global_id(0);

	double4 pos = positions[thread];
	double4 velo = velocities[thread];
  double c_sphere = c_spheres[thread];

  unsigned int num_steps = sim_properties.y;
  unsigned int iter_nth = sim_properties.z;
  unsigned int num_coils = sim_properties.w;

	double4 accel;

  // max num of coils is 6
  float4 local_coils[6];
  for(int i = 0; i<num_coils; i++){
    local_coils[i] = coils[thread*num_coils + i];
  }

	double4 k1, k2, k3, k4, l1, l2, l3, l4;

	for(int iter = 0; iter<(num_steps); iter++){

		for (unsigned int sub_int = 0; sub_int < iter_nth; sub_int++){

			//Runge Kutta 4th Order

			k1 = dt * get_accel(pos, velo, local_coils, num_coils, c_sphere, ee_tab, ek_tab);
			l1 = dt * velo;

			k2 = dt * get_accel(  (pos + (0.5f * l1)), velo, local_coils, num_coils, c_sphere, ee_tab, ek_tab);
			l2 = dt * (velo + (0.5f * k1));

			k3 = dt * get_accel( (pos + (0.5f * l2)), velo, local_coils, num_coils, c_sphere, ee_tab, ek_tab);
			l3 = dt * (velo + (0.5f * k2));

			k4 = dt * get_accel( (pos + l3), velo, local_coils, num_coils, c_sphere, ee_tab, ek_tab);
			l4 = dt * (velo + k3);

			velo += (k1 + (2.0f*k2) + (2.0f*k3) +k4)/6.0f;
			pos += (l1 + (2.0f*l2) + (2.0f*l3) +l4)/6.0f;

      /*
      Euler's Method
			accel = get_accel(pos, velo, local_coils, num_coils, c_sphere, ee_tab, ek_tab); //returns acceleration
			velo += (accel * dt);
			pos += (velo * dt);
      */
		}
		dest[thread*num_steps + iter] = (float4)(pos.x, pos.y, pos.z, accel.w);
	}
}

/*
Device Info:

1 gb Max Global Memory
32 kbytes local memory
64 kbytes constant memory
6 compute units
657 MHz
float4s favored

*/
