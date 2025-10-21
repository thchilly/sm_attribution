function [D, M, PI, DDD, DRD]=runtheo(ssi_c)
ssi_ct=ssi_c;
% plot(ssi_c); hold on;
xp= ssi_c>=0 & ssi_c<=1;
[ev_01p] = bwlabel(xp);
for i=1:max(ev_01p)
    if sum(ev_01p==i)<6
        ssi_ct(ev_01p==i)=-0.000001;
    end
end
xn= ssi_ct<0;
[ev_01n] = bwlabel(xn);
for i=1:max(ev_01n)
    if min(ssi_ct(ev_01n==i))>-1
        ssi_ct(ev_01n==i)=+0.000001;
    end
end
BW=bwlabel(ssi_ct<0);
fiT=[];
for i=1:max(BW)
    D(i)=sum(BW==i);
    fiT(i)=nanmean(find(BW==i));
    fi=find(BW==i);
    evd=ssi_ct(BW==i);
    M(i)=sum(abs(evd));
    PI(i)=M(i)./D(i);
    k=find(evd==min(evd));
    DDD(i)=k(1);
    DRD(i)=length(evd)-DDD(i);
end

if isempty(fiT)
    D=NaN;
    M=NaN;
    PI=NaN;
    DDD=NaN;
    DRD=NaN;
end
end
