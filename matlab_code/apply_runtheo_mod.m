function [cDs,cMs,cPIs,cDDDs,cDRDs]=apply_runtheo_mod(smd_cl_b,smd_oc_a)
cDs=nan(size(smd_cl_b,1), size(smd_cl_b,2));
cMs=nan(size(smd_cl_b,1), size(smd_cl_b,2));
cPIs=nan(size(smd_cl_b,1), size(smd_cl_b,2));
cDDDs=nan(size(smd_cl_b,1), size(smd_cl_b,2));
cDRDs=nan(size(smd_cl_b,1), size(smd_cl_b,2));
        
% ii=353; jj=100;
for ii=1:size(smd_cl_b,1) % 380%
     for jj=1:size(smd_cl_b,2) %100%
        if ~isnan(smd_cl_b(ii,jj,1)) & sum(smd_cl_b(ii,jj,:))~=0 
            inpd_c=permute(smd_cl_b(ii,jj,:),[3 2 1]);
            inpd_o=permute(smd_oc_a(ii,jj,:),[3 2 1]);
           
            [xx, SI_proj]=SSI_mod(inpd_c,inpd_o,3);   % td_ref,td_proj,sc 
            
            if min(SI_proj)<-1
                [cD, cM, cPI, cDDD, cDRD]=runtheo(SI_proj);
                cDs(ii,jj)=nanmean(cD);
                cMs(ii,jj)=nanmean(cM);
                cPIs(ii,jj)=nanmean(cPI);
                cDDDs(ii,jj)=nanmean(cDDD);
                cDRDs(ii,jj)=nanmean(cDRD);
            end
        end
    end
end