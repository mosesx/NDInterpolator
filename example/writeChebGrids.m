function []=writeChebGrids(n)

for k=0:n-1
    G=genGrid([-1.8,-.8],[2.3,3.3],[15,15],2,(k+1)/n,50,.33);
    dlmwrite(['ex2Grid',num2str(k),'.dat'],G','delimiter','\t','precision',16);
end

return