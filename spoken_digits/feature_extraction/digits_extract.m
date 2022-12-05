speakers = ["george" "jackson" "lucas" "nicolas" "theo" "yweweler"];
rec_per_speaker = 50;
digits_path = "../wav_digits/";
mfcc_path = "../txt_mfccs/";
lpcc_path = "../txt_lpccs/";

for speaker = 1:length(speakers)
    for digit = 1:10
        for rec_num = 1:rec_per_speaker
            name = int2str(digit - 1) + "_" + speakers(speaker) + "_" + int2str(rec_num - 1);
            [mfcc, lpcc] = mfcc_lpcc_extract(digits_path + name + ".wav");
            writematrix(mfcc, mfcc_path + name + "_mfcc.txt")
            writematrix(lpcc, lpcc_path + name + "_lpcc.txt")
        end
    end
end