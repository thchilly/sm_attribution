function smonemet=fix_cwat(smd_cl)

L1sm=squeeze(smd_cl(:,:,1,:));
L2sm=squeeze(smd_cl(:,:,2,:));
L3sm=squeeze(smd_cl(:,:,3,:));

fracforest=ncread([pwd,'\CWATfiles\fractionLandcover_1850_2018.nc'],'fracforest');
fracgrassland=ncread([pwd,'\CWATfiles\fractionLandcover_1850_2018.nc'],'fracgrassland');
fracirrPaddy=ncread([pwd,'\CWATfiles\fractionLandcover_1850_2018.nc'],'fracirrPaddy');
fracirrNonPaddy=ncread([pwd,'\CWATfiles\fractionLandcover_1850_2018.nc'],'fracirrNonPaddy');

maxRootDepth_forest=ncread([pwd,'\CWATfiles\maxRootDepth_forest.nc'],'maxRootDepth');
maxRootDepth_grass=ncread([pwd,'\CWATfiles\maxRootDepth_grass.nc'],'maxRootDepth');
maxRootDepth_irrPaddy=ncread([pwd,'\CWATfiles\maxRootDepth_irrPaddy.nc'],'maxRootDepth');
maxRootDepth_irrNonPaddy=ncread([pwd,'\CWATfiles\maxRootDepth_irrNonPaddy.nc'],'maxRootDepth');

Depthlayer2=ncread([pwd,'\CWATfiles\storageDepth1.nc'],'firstStorDepth');
Depthlayer3=ncread([pwd,'\CWATfiles\storageDepth2.nc'],'secondStorDepth');

dl2=max(ones(size(Depthlayer2)).*0.05,Depthlayer2-0.05);
dl3=max(ones(size(Depthlayer2)).*0.05,Depthlayer3);

L2thick_for = min(dl2+dl3-0.05,max(dl2,maxRootDepth_forest- 0.05));
L2thick_irrPaddy = min(dl2+dl3-0.05,max(dl2,maxRootDepth_irrPaddy- 0.05));
L2thick_irrNonPaddy = min(dl2+dl3-0.05,max(dl2,maxRootDepth_irrNonPaddy- 0.05));

L3thick_for = max(ones(size(Depthlayer2)).*0.05,dl2+dl3-L2thick_for);
L3thick_irrPaddy = max(ones(size(Depthlayer2)).*0.05,dl2+dl3-L2thick_irrPaddy);
L3thick_irrNonPaddy = max(ones(size(Depthlayer2)).*0.05,dl2+dl3-L2thick_irrNonPaddy);

smonemet_grass=nan(size(L1sm,1),size(L1sm,2),size(L1sm,3));
smonemet_for=nan(size(L1sm,1),size(L1sm,2),size(L1sm,3));
smonemet_pad=nan(size(L1sm,1),size(L1sm,2),size(L1sm,3));
smonemet_nop=nan(size(L1sm,1),size(L1sm,2),size(L1sm,3));
for yr=1:119
    yr
    L1sm_y=L1sm(:,:,yr*12-11:yr*12);
    L2sm_y=L2sm(:,:,yr*12-11:yr*12);
    L3sm_y=L3sm(:,:,yr*12-11:yr*12);
    for ii=1:720
        for jj=1:360
            smonemet_tg(ii,jj,:)=fracgrassland(ii,jj,yr).*(L1sm_y(ii,jj,:) + L2sm_y(ii,jj,:))./(maxRootDepth_grass(ii,jj,:));
            if L2thick_for(ii,jj,:)==0.95
                smonemet_tf(ii,jj,:)=fracforest(ii,jj,yr).*L1sm_y(ii,jj,:) + fracforest(ii,jj,yr).*L2sm_y(ii,jj);
            else
                smonemet_tf(ii,jj,:)=fracforest(ii,jj,yr).*(L1sm_y(ii,jj,:) + L2sm_y(ii,jj) + L3sm_y(ii,jj,:).*((1-L2thick_for(ii,jj,:)+0.05)./L3thick_for(ii,jj,:)));
            end
            smonemet_tp(ii,jj,:)=fracirrPaddy(ii,jj,yr).*(L1sm_y(ii,jj,:).*0.05 + L2sm_y(ii,jj,:) + L3sm_y(ii,jj,:).*((1-L2thick_irrPaddy(ii,jj,:)+0.05)./L3thick_irrPaddy(ii,jj,:)));
            if L2thick_irrNonPaddy(ii,jj,:)==0.95
                smonemet_tn(ii,jj,:)=fracirrNonPaddy(ii,jj,yr).*L1sm_y(ii,jj,:) + fracirrNonPaddy(ii,jj,yr).*L2sm_y(ii,jj);
            else
                smonemet_tn(ii,jj,:)=fracirrNonPaddy(ii,jj,yr).*(L1sm_y(ii,jj,:) + L2sm_y(ii,jj) + L3sm_y(ii,jj,:).*((1-L2thick_irrNonPaddy(ii,jj,:)+0.05)./L3thick_irrNonPaddy(ii,jj,:)));
            end
        end
    end
    smonemet_grass(:,:,yr*12-11:yr*12)=smonemet_tg;
    smonemet_for(:,:,yr*12-11:yr*12)=smonemet_tf;
    smonemet_pad(:,:,yr*12-11:yr*12)=smonemet_tp;
    smonemet_nop(:,:,yr*12-11:yr*12)=smonemet_tn;
end
smonemet=smonemet_grass+smonemet_for+smonemet_pad+smonemet_nop;

%  imshow(nanmean(smonemet,3)',[]); colormap jet;
% imshow(fracforest(:,:,1)'+fracgrassland(:,:,1)'+fracirrPaddy(:,:,1)'+fracirrNonPaddy(:,:,1)',[]); colormap jet;