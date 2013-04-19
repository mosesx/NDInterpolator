function []=ublasMatrixIO(A,filename)

[n,m]=size(A);

fid=fopen(filename,'w');
fprintf(fid,['[',num2str(n),',',num2str(m),'](']);

for k=1:n
    fprintf(fid,'(');
    for l=1:m-1
        fprintf(fid,[num2str(A(k,l),'%.16e'),',']);
    end
    if k<n
        fprintf(fid,num2str(A(k,m),'%.16e'));
        fprintf(fid,'),');
    else
        fprintf(fid,num2str(A(k,m),'%.16e'));
        fprintf(fid,')'); % no comma if last line reached
    end
end
fprintf(fid,')');
fclose(fid);

return