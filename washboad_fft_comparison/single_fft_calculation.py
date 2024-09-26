profile = self.y_cuts[i]
profile_offset = np.mean(profile[1]) 
profile[1] = profile[1]- profile_offset # Align the profile data with zero on the y-axis

    # Compute FFT
time_values = profile[0]/1000 # Needs to be divided to obtain same as test file
dt = np.mean(np.diff(time_values))  # Compute the average time step
# Perform FFT on experimental data
fft_result_exp = np.fft.fft(profile[1])*dt
fft_freq_exp = np.fft.fftfreq(len(profile[1]), d=dt)*dt
output_file_data_fft = os.path.join(filename,'fft_'+str(i+6)+'th.txt')
np.savetxt(output_file_data_fft, [fft_freq_exp, np.abs(fft_result_exp)], fmt='%.6f', delimiter='\n')