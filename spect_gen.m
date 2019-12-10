clear all; close all;
dir = dir('/Users/chenshi/Downloads/Accent-Classification-Using-Signals-master/audio');
cd /Users/chenshi/Downloads/Accent-Classification-Using-Signals-master/abs_spect
max_log = 53.06;
min_log = max_log-80;
color_map = parula(256);

for i = 3:length(dir)
    y = zeros(64*3,12160+1);
    file_name = dir(i).name;
    [file, fs] = audioread(['/Users/chenshi/Downloads/Accent-Classification-Using-Signals-master/audio/' file_name]);
    file_spect = abs(spectrogram(file,fs*0.03,round(fs*0.03*3/4),126));
    log_spect = 20*log10(file_spect);
    [r,c] = size(log_spect);
    num_img = ceil(c/64);
    x_log = zeros(64,12160) - 40;
    x_log(1:r,1:c) = log_spect;
    x_eq = zeros(64,12160);
    for m = 1:64
        for n = 1:12160
            if x_log(m,n) < min_log
                x_eq(m,n) = 0;
            else
                x_eq(m,n) = round((x_log(m,n)-min_log)*256/80)-1;
            end
            for l=1:256
                 if x_eq(m,n) == l-1
                    y(m,n) = color_map(l,1);
                    y(m+64,n) = color_map(l,2);
                    y(m+128,n) = color_map(l,3);
                 end
            end
        end
    end% equaliztion
    y(:,12161) = num_img;
    file_name_new = [file_name(1:end-4) '_spectrum.csv'];
    csvwrite(file_name_new,y);
end