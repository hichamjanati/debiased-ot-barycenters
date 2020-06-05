global max_sample_size s_modalities d_modalities base_part rand_part;
N = max_sample_size*s_modalities;
m = 1000;

fid=fopen('test.d2', 'wt');
for i = 1:N
    mt = int64(rand()*rand_part+base_part);
    fprintf(fid, '%d\n %d\n', d_modalities, mt);
    w = rand(1,mt)+5e-3;
    w = w/sum(w);
    for j = 1:mt
        fprintf(fid, '%4.5f ', w(j));
    end
    fprintf(fid, '\n ');
    loc = rand(3,1);
    for j = 1:mt
        wt =  randn(1,d_modalities)*100;
        for k = 1:d_modalities
            fprintf(fid, '%4.6f ', wt(k));
        end
        fprintf(fid, '\n ');
    end
    fprintf(fid, '\n ');
end
fclose(fid);