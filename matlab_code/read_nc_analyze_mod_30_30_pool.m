clear; clc;

resultsfol='asciis_30_30g_pool';
datafol='I:\OneDrive\z_ISIMIP3a_analysis\aa_INPUT\';
                               
modlnam=[{'h08'},{'hydropy'},{'jules-w2'},{'miroc-integ-land'},{'watergap2-2e'},{'web-dhm-sg'}]; % Sta MEDIANS exo afairesei to teleytaio montelo, dld 
varnam=[{'soilmoist'},{'rootmoist'},{'soilmoist'},{'soilmoist'},{'soilmoist'},{'soilmoist'}];

tic
cc=1;
smln=length(modlnam);

maska=xlsread('I:\OneDrive\z_ISIMIP3a_analysis\koppen_mask\Koppen.xlsx','map');
maska(maska==-9999)=NaN;
maska(maska~=30)=1;
maska(maska~=1)=NaN;

for iio=1:length(modlnam)
    iio
    lst=dir(['I:\OneDrive\z_ISIMIP3a_analysis\aa_INPUT\',modlnam{iio},'*counterclim*']);
    smd_cl=ncread(['I:\OneDrive\z_ISIMIP3a_analysis\aa_INPUT\',lst.name],varnam{iio});
    smd_cl=squeeze(smd_cl);

    if strcmp(modlnam{iio},'classic')
        smd_cl=squeeze(nansum(smd_cl(:,:,1:3,:),3));
    end
    if strcmp(modlnam{iio},'miroc-integ-land')
        smd_cl=squeeze(smd_cl(:,:,1,:)+smd_cl(:,:,2,:)+smd_cl(:,:,3,:));
    end
    if strcmp(modlnam{iio},'cwatm')
        smd_cl=fix_cwat(smd_cl);
    end
    if strcmp(modlnam{iio},'jules-w2')
        smd_cl=squeeze(nansum(smd_cl(:,:,1:3,:),3));
        smd_cl(:,:,end)=smd_cl(:,:,end-1);
    end
    if strcmp(modlnam{iio},'ssib4-triffid-fire')
        smd_cl=squeeze(smd_cl(:,:,1,:)+smd_cl(:,:,2,:)+smd_cl(:,:,3,:).*((1-0.48)./(2-0.48)));
    end
%________________________
    
    lst2=dir(['I:\OneDrive\z_ISIMIP3a_analysis\aa_INPUT\',modlnam{iio},'*obsclim*']);
    smd_oc=ncread(['I:\OneDrive\z_ISIMIP3a_analysis\aa_INPUT\',lst2.name],varnam{iio});
    smd_oc=squeeze(smd_oc);
    if strcmp(modlnam{iio},'classic')
        smd_oc=squeeze(nansum(smd_oc(:,:,1:3,:),3));
    end
    if strcmp(modlnam{iio},'miroc-integ-land')
        smd_oc=squeeze(smd_oc(:,:,1,:)+smd_oc(:,:,2,:)+smd_oc(:,:,3,:));
    end
    if strcmp(modlnam{iio},'cwatm')
        smd_oc=fix_cwat(smd_oc);
    end
    if strcmp(modlnam{iio},'jules-w2')
        smd_oc=squeeze(nansum(smd_oc(:,:,1:3,:),3));
        smd_oc(:,:,end)=smd_oc(:,:,end-1);
    end
    if strcmp(modlnam{iio},'ssib4-triffid-fire')
        smd_oc=squeeze(smd_oc(:,:,1,:)+smd_oc(:,:,2,:)+smd_oc(:,:,3,:).*((1-0.48)./(2-0.48)));
    end
%     imshow(nanmean(smd_oc,3)',[0 100]); colormap jet
    %________________________1901soc_OC
    lst3=dir(['H:\ISIMIP3_1901_soc\',modlnam{iio},'*obsclim*']);
    smd_oc19=ncread(['H:\ISIMIP3_1901_soc\',lst3.name],varnam{iio});
    smd_oc19=squeeze(smd_oc19);
    if strcmp(modlnam{iio},'classic')
        smd_oc19=squeeze(nansum(smd_oc19(:,:,1:3,:),3));
    end
    if strcmp(modlnam{iio},'miroc-integ-land')
        smd_oc19=squeeze(smd_oc19(:,:,1,:)+smd_oc19(:,:,2,:)+smd_oc19(:,:,3,:));
    end
    if strcmp(modlnam{iio},'cwatm')
        smd_oc19=fix_cwat(smd_oc19);
    end
    if strcmp(modlnam{iio},'jules-w2')
        smd_oc19=squeeze(nansum(smd_oc19(:,:,1:3,:),3));
        smd_oc19(:,:,end)=smd_oc19(:,:,end-1);
    end
    if strcmp(modlnam{iio},'ssib4-triffid-fire')
        smd_oc19=squeeze(smd_oc19(:,:,1,:)+smd_oc19(:,:,2,:)+smd_oc19(:,:,3,:).*((1-0.48)./(2-0.48)));
    end
    smd_cl=squeeze(smd_cl);
    smd_oc=squeeze(smd_oc);
    smd_oc19=squeeze(smd_oc19);

%________________________1901soc_CC
    lst4=dir(['H:\ISIMIP3_1901_soc\',modlnam{iio},'*counterclim*']);
    smd_cl19=ncread(['H:\ISIMIP3_1901_soc\',lst4.name],varnam{iio});
    smd_cl19=squeeze(smd_cl19);
    if strcmp(modlnam{iio},'classic')
        smd_cl19=squeeze(nansum(smd_cl19(:,:,1:3,:),3));
    end
    if strcmp(modlnam{iio},'miroc-integ-land')
        smd_cl19=squeeze(smd_cl19(:,:,1,:)+smd_cl19(:,:,2,:)+smd_cl19(:,:,3,:));
    end
    if strcmp(modlnam{iio},'cwatm')
        smd_cl19=fix_cwat(smd_cl19);
    end
    if strcmp(modlnam{iio},'jules-w2')
        smd_cl19=squeeze(nansum(smd_cl19(:,:,1:3,:),3));
        smd_cl19(:,:,end)=smd_cl19(:,:,end-1);
    end
    if strcmp(modlnam{iio},'ssib4-triffid-fire')
        smd_cl19=squeeze(smd_cl19(:,:,1,:)+smd_cl19(:,:,2,:)+smd_cl19(:,:,3,:).*((1-0.48)./(2-0.48)));
    end

    smd_cl=squeeze(smd_cl);
    smd_oc=squeeze(smd_oc);
    smd_cl19=squeeze(smd_cl19);
    smd_oc19=squeeze(smd_oc19);

    datapool=cat(3,smd_cl,smd_oc,smd_oc19,smd_cl19);
    [Dcl_b30(:,:,cc), Mcl_b30(:,:,cc), PIcl_b30(:,:,cc), DDDcl_b30(:,:,cc), DRDcl_b30(:,:,cc)]=apply_runtheo_mod(datapool,smd_cl(:,:,1:360));
    [Dcl_a30(:,:,cc), Mcl_a30(:,:,cc), PIcl_a30(:,:,cc), DDDcl_a30(:,:,cc), DRDcl_a30(:,:,cc)]=apply_runtheo_mod(datapool,smd_cl(:,:,end-359:end));
    [Doc_b30(:,:,cc), Moc_b30(:,:,cc), PIoc_b30(:,:,cc), DDDoc_b30(:,:,cc), DRDoc_b30(:,:,cc)]=apply_runtheo_mod(datapool,smd_oc(:,:,1:360));
    [Doc_a30(:,:,cc), Moc_a30(:,:,cc), PIoc_a30(:,:,cc), DDDoc_a30(:,:,cc), DRDoc_a30(:,:,cc)]=apply_runtheo_mod(datapool,smd_oc(:,:,end-359:end));
    
    [Dcl_b30_19(:,:,cc), Mcl_b30_19(:,:,cc), PIcl_b30_19(:,:,cc), DDDcl_b30_19(:,:,cc), DRDcl_b30_19(:,:,cc)]=apply_runtheo_mod(datapool,smd_cl19(:,:,1:360));
    [Dcl_a30_19(:,:,cc), Mcl_a30_19(:,:,cc), PIcl_a30_19(:,:,cc), DDDcl_a30_19(:,:,cc), DRDcl_a30_19(:,:,cc)]=apply_runtheo_mod(datapool,smd_cl19(:,:,end-359:end));
    [Doc_b30_19(:,:,cc), Moc_b30_19(:,:,cc), PIoc_b30_19(:,:,cc), DDDoc_b30_19(:,:,cc), DRDoc_b30_19(:,:,cc)]=apply_runtheo_mod(datapool,smd_oc19(:,:,1:360));
    [Doc_a30_19(:,:,cc), Moc_a30_19(:,:,cc), PIoc_a30_19(:,:,cc), DDDoc_a30_19(:,:,cc), DRDoc_a30_19(:,:,cc)]=apply_runtheo_mod(datapool,smd_oc19(:,:,end-359:end));
    cc=cc+1
end % Intensity_factual_RP_1901soc
%%
Duration_mask=sum(~isnan(Dcl_b30),3)>2 | sum(~isnan(Doc_b30),3)>2 | sum(~isnan(Dcl_a30),3)>2 | sum(~isnan(Doc_a30),3)>2; 
Magnitude_mask=sum(~isnan(Mcl_b30),3)>2 | sum(~isnan(Moc_b30),3)>2 | sum(~isnan(Mcl_a30),3)>2 | sum(~isnan(Moc_a30),3)>2;
Intensity_mask=sum(~isnan(PIcl_b30),3)>2 | sum(~isnan(PIoc_b30),3)>2 | sum(~isnan(PIcl_a30),3)>2 | sum(~isnan(PIoc_a30),3)>2;
DDD_mask=sum(~isnan(DDDcl_b30),3)>2 | sum(~isnan(DDDoc_b30),3)>2 | sum(~isnan(DDDcl_a30),3)>2 | sum(~isnan(DDDoc_a30),3)>2;
DRD_mask=sum(~isnan(DRDcl_b30),3)>2 | sum(~isnan(DRDoc_b30),3)>2 | sum(~isnan(DRDcl_a30),3)>2 | sum(~isnan(DRDoc_a30),3)>2; 

Duration_mask=Duration_mask.*maska';
Magnitude_mask=Magnitude_mask.*maska';
Intensity_mask=Intensity_mask.*maska';
DDD_mask=DDD_mask.*maska';
DRD_mask=DRD_mask.*maska';

Dcl_b30=repmat(Duration_mask,1,1,smln).*Dcl_b30;
Mcl_b30=repmat(Magnitude_mask,1,1,smln).*Mcl_b30;
PIcl_b30=repmat(Intensity_mask,1,1,smln).*PIcl_b30;
DDDcl_b30=repmat(DDD_mask,1,1,smln).*DDDcl_b30;
DRDcl_b30=repmat(DRD_mask,1,1,smln).*DRDcl_b30;

Doc_b30=repmat(Duration_mask,1,1,smln).*Doc_b30;
Moc_b30=repmat(Magnitude_mask,1,1,smln).*Moc_b30;
PIoc_b30=repmat(Intensity_mask,1,1,smln).*PIoc_b30;
DDDoc_b30=repmat(DDD_mask,1,1,smln).*DDDoc_b30;
DRDoc_b30=repmat(DRD_mask,1,1,smln).*DRDoc_b30;

Dcl_a30=repmat(Duration_mask,1,1,smln).*Dcl_a30;
Mcl_a30=repmat(Magnitude_mask,1,1,smln).*Mcl_a30;
PIcl_a30=repmat(Intensity_mask,1,1,smln).*PIcl_a30;
DDDcl_a30=repmat(DDD_mask,1,1,smln).*DDDcl_a30;
DRDcl_a30=repmat(DRD_mask,1,1,smln).*DRDcl_a30;

Doc_a30=repmat(Duration_mask,1,1,smln).*Doc_a30;
Moc_a30=repmat(Magnitude_mask,1,1,smln).*Moc_a30;
PIoc_a30=repmat(Intensity_mask,1,1,smln).*PIoc_a30;
DDDoc_a30=repmat(DDD_mask,1,1,smln).*DDDoc_a30;
DRDoc_a30=repmat(DRD_mask,1,1,smln).*DRDoc_a30;
%
Duration_mask_19=sum(~isnan(Dcl_b30_19),3)>2 | sum(~isnan(Doc_b30_19),3)>2 | sum(~isnan(Dcl_a30_19),3)>2 | sum(~isnan(Doc_a30_19),3)>2; 
Magnitude_mask_19=sum(~isnan(Mcl_b30_19),3)>2 | sum(~isnan(Moc_b30_19),3)>2 | sum(~isnan(Mcl_a30_19),3)>2 | sum(~isnan(Moc_a30_19),3)>2;
Intensity_mask_19=sum(~isnan(PIcl_b30_19),3)>2 | sum(~isnan(PIoc_b30_19),3)>2 | sum(~isnan(PIcl_a30_19),3)>2 | sum(~isnan(PIoc_a30_19),3)>2;
DDD_mask_19=sum(~isnan(DDDcl_b30_19),3)>2 | sum(~isnan(DDDoc_b30_19),3)>2 | sum(~isnan(DDDcl_a30_19),3)>2 | sum(~isnan(DDDoc_a30_19),3)>2;
DRD_mask_19=sum(~isnan(DRDcl_b30_19),3)>2 | sum(~isnan(DRDoc_b30_19),3)>2 | sum(~isnan(DRDcl_a30_19),3)>2 | sum(~isnan(DRDoc_a30_19),3)>2; 

Duration_mask_19=Duration_mask_19.*maska';
Magnitude_mask_19=Magnitude_mask_19.*maska';
Intensity_mask_19=Intensity_mask_19.*maska';
DDD_mask_19=DDD_mask_19.*maska';
DRD_mask_19=DRD_mask_19.*maska';

Doc_b30_19=repmat(Duration_mask_19,1,1,smln).*Doc_b30_19;
Moc_b30_19=repmat(Magnitude_mask_19,1,1,smln).*Moc_b30_19;
PIoc_b30_19=repmat(Intensity_mask_19,1,1,smln).*PIoc_b30_19;
DDDoc_b30_19=repmat(DDD_mask_19,1,1,smln).*DDDoc_b30_19;
DRDoc_b30_19=repmat(DRD_mask_19,1,1,smln).*DRDoc_b30_19;

Doc_a30_19=repmat(Duration_mask_19,1,1,smln).*Doc_a30_19;
Moc_a30_19=repmat(Magnitude_mask_19,1,1,smln).*Moc_a30_19;
PIoc_a30_19=repmat(Intensity_mask_19,1,1,smln).*PIoc_a30_19; %%
DDDoc_a30_19=repmat(DDD_mask_19,1,1,smln).*DDDoc_a30_19;
DRDoc_a30_19=repmat(DRD_mask_19,1,1,smln).*DRDoc_a30_19;

for x=1:size(Dcl_b30,3)

    write_ascii([pwd,'\',resultsfol,'\',modlnam{x},'_counterfact_Duration_EI_histsoc.asc'],Dcl_b30(:,:,x)',720,360,-180,-90,0.5);
    write_ascii([pwd,'\',resultsfol,'\',modlnam{x},'_counterfact_Magnitude_EI_histsoc.asc'],Mcl_b30(:,:,x)',720,360,-180,-90,0.5);
    write_ascii([pwd,'\',resultsfol,'\',modlnam{x},'_counterfact_Intensity_EI_histsoc.asc'],PIcl_b30(:,:,x)',720,360,-180,-90,0.5);
    write_ascii([pwd,'\',resultsfol,'\',modlnam{x},'_counterfact_DDD_EI_histsoc.asc'],DDDcl_b30(:,:,x)',720,360,-180,-90,0.5);
    write_ascii([pwd,'\',resultsfol,'\',modlnam{x},'_counterfact_DRD_EI_histsoc.asc'],DRDcl_b30(:,:,x)',720,360,-180,-90,0.5);
    
    write_ascii([pwd,'\',resultsfol,'\',modlnam{x},'_factual_Duration_EI_histsoc.asc'],Doc_b30(:,:,x)',720,360,-180,-90,0.5);
    write_ascii([pwd,'\',resultsfol,'\',modlnam{x},'_factual_Magnitude_EI_histsoc.asc'],Moc_b30(:,:,x)',720,360,-180,-90,0.5);
    write_ascii([pwd,'\',resultsfol,'\',modlnam{x},'_factual_Intensity_EI_histsoc.asc'],PIoc_b30(:,:,x)',720,360,-180,-90,0.5);
    write_ascii([pwd,'\',resultsfol,'\',modlnam{x},'_factual_DDD_EI_histsoc.asc'],DDDoc_b30(:,:,x)',720,360,-180,-90,0.5);
    write_ascii([pwd,'\',resultsfol,'\',modlnam{x},'_factual_DRD_EI_histsoc.asc'],DRDoc_b30(:,:,x)',720,360,-180,-90,0.5);
    
    write_ascii([pwd,'\',resultsfol,'\',modlnam{x},'_counterfact_Duration_RP_histsoc.asc'],Dcl_a30(:,:,x)',720,360,-180,-90,0.5);
    write_ascii([pwd,'\',resultsfol,'\',modlnam{x},'_counterfact_Magnitude_RP_histsoc.asc'],Mcl_a30(:,:,x)',720,360,-180,-90,0.5);
    write_ascii([pwd,'\',resultsfol,'\',modlnam{x},'_counterfact_Intensity_RP_histsoc.asc'],PIcl_a30(:,:,x)',720,360,-180,-90,0.5);
    write_ascii([pwd,'\',resultsfol,'\',modlnam{x},'_counterfact_DDD_RP_histsoc.asc'],DDDcl_a30(:,:,x)',720,360,-180,-90,0.5);
    write_ascii([pwd,'\',resultsfol,'\',modlnam{x},'_counterfact_DRD_RP_histsoc.asc'],DRDcl_a30(:,:,x)',720,360,-180,-90,0.5);
    % 
    write_ascii([pwd,'\',resultsfol,'\',modlnam{x},'_factual_Duration_RP_histsoc.asc'],Doc_a30(:,:,x)',720,360,-180,-90,0.5);
    write_ascii([pwd,'\',resultsfol,'\',modlnam{x},'_factual_Magnitude_RP_histsoc.asc'],Moc_a30(:,:,x)',720,360,-180,-90,0.5);
    write_ascii([pwd,'\',resultsfol,'\',modlnam{x},'_factual_Intensity_RP_histsoc.asc'],PIoc_a30(:,:,x)',720,360,-180,-90,0.5);
    write_ascii([pwd,'\',resultsfol,'\',modlnam{x},'_factual_DDD_RP_histsoc.asc'],DDDoc_a30(:,:,x)',720,360,-180,-90,0.5);
    write_ascii([pwd,'\',resultsfol,'\',modlnam{x},'_factual_DRD_RP_histsoc.asc'],DRDoc_a30(:,:,x)',720,360,-180,-90,0.5);
%
    write_ascii([pwd,'\',resultsfol,'\',modlnam{x},'_counterfact_Duration_EI_1901soc.asc'],Dcl_b30_19(:,:,x)',720,360,-180,-90,0.5);
    write_ascii([pwd,'\',resultsfol,'\',modlnam{x},'_counterfact_Magnitude_EI_1901soc.asc'],Mcl_b30_19(:,:,x)',720,360,-180,-90,0.5);
    write_ascii([pwd,'\',resultsfol,'\',modlnam{x},'_counterfact_Intensity_EI_1901soc.asc'],PIcl_b30_19(:,:,x)',720,360,-180,-90,0.5);
    write_ascii([pwd,'\',resultsfol,'\',modlnam{x},'_counterfact_DDD_EI_1901soc.asc'],DDDcl_b30_19(:,:,x)',720,360,-180,-90,0.5);
    write_ascii([pwd,'\',resultsfol,'\',modlnam{x},'_counterfact_DRD_EI_1901soc.asc'],DRDcl_b30_19(:,:,x)',720,360,-180,-90,0.5);

    write_ascii([pwd,'\',resultsfol,'\',modlnam{x},'_factual_Duration_EI_1901soc.asc'],Doc_b30_19(:,:,x)',720,360,-180,-90,0.5);
    write_ascii([pwd,'\',resultsfol,'\',modlnam{x},'_factual_Magnitude_EI_1901soc.asc'],Moc_b30_19(:,:,x)',720,360,-180,-90,0.5);
    write_ascii([pwd,'\',resultsfol,'\',modlnam{x},'_factual_Intensity_EI_1901soc.asc'],PIoc_b30_19(:,:,x)',720,360,-180,-90,0.5);
    write_ascii([pwd,'\',resultsfol,'\',modlnam{x},'_factual_DDD_EI_1901soc.asc'],DDDoc_b30_19(:,:,x)',720,360,-180,-90,0.5);
    write_ascii([pwd,'\',resultsfol,'\',modlnam{x},'_factual_DRD_EI_1901soc.asc'],DRDoc_b30_19(:,:,x)',720,360,-180,-90,0.5);

    write_ascii([pwd,'\',resultsfol,'\',modlnam{x},'_counterfact_Duration_RP_1901soc.asc'],Dcl_a30_19(:,:,x)',720,360,-180,-90,0.5);
    write_ascii([pwd,'\',resultsfol,'\',modlnam{x},'_counterfact_Magnitude_RP_1901soc.asc'],Mcl_a30_19(:,:,x)',720,360,-180,-90,0.5);
    write_ascii([pwd,'\',resultsfol,'\',modlnam{x},'_counterfact_Intensity_RP_1901soc.asc'],PIcl_a30_19(:,:,x)',720,360,-180,-90,0.5);
    write_ascii([pwd,'\',resultsfol,'\',modlnam{x},'_counterfact_DDD_RP_1901soc.asc'],DDDcl_a30_19(:,:,x)',720,360,-180,-90,0.5);
    write_ascii([pwd,'\',resultsfol,'\',modlnam{x},'_counterfact_DRD_RP_1901soc.asc'],DRDcl_a30_19(:,:,x)',720,360,-180,-90,0.5);
    
    write_ascii([pwd,'\',resultsfol,'\',modlnam{x},'_factual_Duration_RP_1901soc.asc'],Doc_a30_19(:,:,x)',720,360,-180,-90,0.5);
    write_ascii([pwd,'\',resultsfol,'\',modlnam{x},'_factual_Magnitude_RP_1901soc.asc'],Moc_a30_19(:,:,x)',720,360,-180,-90,0.5);
    write_ascii([pwd,'\',resultsfol,'\',modlnam{x},'_factual_Intensity_RP_1901soc.asc'],PIoc_a30_19(:,:,x)',720,360,-180,-90,0.5);
    write_ascii([pwd,'\',resultsfol,'\',modlnam{x},'_factual_DDD_RP_1901soc.asc'],DDDoc_a30_19(:,:,x)',720,360,-180,-90,0.5);
    write_ascii([pwd,'\',resultsfol,'\',modlnam{x},'_factual_DRD_RP_1901soc.asc'],DRDoc_a30_19(:,:,x)',720,360,-180,-90,0.5);
%
    write_ascii([pwd,'\',resultsfol,'\',modlnam{x},'_diff_Duration_factual_RP_histsoc-counterfact_RP_histsoc.asc'],Doc_a30(:,:,x)'-Dcl_a30(:,:,x)',720,360,-180,-90,0.5);
    write_ascii([pwd,'\',resultsfol,'\',modlnam{x},'_diff_Magnitude_factual_RP_histsoc-counterfact_RP_histsoc.asc'],Moc_a30(:,:,x)'-Mcl_a30(:,:,x)',720,360,-180,-90,0.5);
    write_ascii([pwd,'\',resultsfol,'\',modlnam{x},'_diff_Intensity_factual_RP_histsoc-counterfact_RP_histsoc.asc'],PIoc_a30(:,:,x)'-PIcl_a30(:,:,x)',720,360,-180,-90,0.5);
    write_ascii([pwd,'\',resultsfol,'\',modlnam{x},'_diff_DDD_factual_RP_histsoc-counterfact_RP_histsoc.asc'],DDDoc_a30(:,:,x)'-DDDcl_a30(:,:,x)',720,360,-180,-90,0.5);
    write_ascii([pwd,'\',resultsfol,'\',modlnam{x},'_diff_DRD_factual_RP_histsoc-counterfact_RP_histsoc.asc'],DRDoc_a30(:,:,x)'-DRDcl_a30(:,:,x)',720,360,-180,-90,0.5);

    write_ascii([pwd,'\',resultsfol,'\',modlnam{x},'_diff_Duration_factual_RP_histsoc-counterfact_EI_1901soc.asc'],Dcl_a30(:,:,x)'-Dcl_b30_19(:,:,x)',720,360,-180,-90,0.5);
    write_ascii([pwd,'\',resultsfol,'\',modlnam{x},'_diff_Magnitude_factual_RP_histsoc-counterfact_EI_1901soc.asc'],Mcl_a30(:,:,x)'-Mcl_b30_19(:,:,x)',720,360,-180,-90,0.5);
    write_ascii([pwd,'\',resultsfol,'\',modlnam{x},'_diff_Intensity_factual_RP_histsoc-counterfact_EI_1901soc.asc'],PIcl_a30(:,:,x)'-PIcl_b30_19(:,:,x)',720,360,-180,-90,0.5);
    write_ascii([pwd,'\',resultsfol,'\',modlnam{x},'_diff_DDD_factual_RP_histsoc-counterfact_RP_EI_1901soc.asc'],DDDcl_a30(:,:,x)'-DDDcl_b30_19(:,:,x)',720,360,-180,-90,0.5);
    write_ascii([pwd,'\',resultsfol,'\',modlnam{x},'_diff_DRD_factual_RP_histsoc-counterfact_RP_EI_1901soc.asc'],DRDcl_a30(:,:,x)'-DRDcl_b30_19(:,:,x)',720,360,-180,-90,0.5);

    write_ascii([pwd,'\',resultsfol,'\',modlnam{x},'_diff_Duration_factual_RP_histsoc-counterfact_EI_1901soc.asc'],Doc_a30(:,:,x)'-Dcl_b30_19(:,:,x)',720,360,-180,-90,0.5);
    write_ascii([pwd,'\',resultsfol,'\',modlnam{x},'_diff_Magnitude_factual_RP_histsoc-counterfact_EI_1901soc.asc'],Moc_a30(:,:,x)'-Mcl_b30_19(:,:,x)',720,360,-180,-90,0.5);
    write_ascii([pwd,'\',resultsfol,'\',modlnam{x},'_diff_Intensity_factual_RP_histsoc-counterfact_EI_1901soc.asc'],PIoc_a30(:,:,x)'-PIcl_b30_19(:,:,x)',720,360,-180,-90,0.5);
    write_ascii([pwd,'\',resultsfol,'\',modlnam{x},'_diff_DDD_factual_RP_histsoc-counterfact_RP_EI_1901soc.asc'],DDDoc_a30(:,:,x)'-DDDcl_b30_19(:,:,x)',720,360,-180,-90,0.5);
    write_ascii([pwd,'\',resultsfol,'\',modlnam{x},'_diff_DRD_factual_RP_histsoc-counterfact_RP_EI_1901soc.asc'],DRDoc_a30(:,:,x)'-DRDcl_b30_19(:,:,x)',720,360,-180,-90,0.5);
end

write_ascii([pwd,'\',resultsfol,'\median_Duration_diff_factual_RP_histsoc-counterfact_RP_histsoc.asc'],nanmedian(Doc_a30-Dcl_a30,3)',720,360,-180,-90,0.5);
write_ascii([pwd,'\',resultsfol,'\median_Magnitude_diff_factual_RP_histsoc-counterfact_RP_histsoc.asc'],nanmedian(Moc_a30-Mcl_a30,3)',720,360,-180,-90,0.5);
write_ascii([pwd,'\',resultsfol,'\median_Intensit_diff_factual_RP_histsoc-counterfact_RP_histsoc.asc'],nanmedian(PIoc_a30-PIcl_a30,3)',720,360,-180,-90,0.5);
write_ascii([pwd,'\',resultsfol,'\median_DDD_diff_factual_RP_histsoc-counterfact_RP_histsoc.asc'],nanmedian(DDDoc_a30-DDDcl_a30,3)',720,360,-180,-90,0.5);
write_ascii([pwd,'\',resultsfol,'\median_DRD_diff_factual_RP_histsoc-counterfact_RP_histsoc.asc'],nanmedian(DRDoc_a30-DRDcl_a30,3)',720,360,-180,-90,0.5);

write_ascii([pwd,'\',resultsfol,'\median_Duration_diff_counterfact_RP_histsoc-counterfact_EI_1901soc.asc'],nanmedian(Dcl_a30-Dcl_b30_19,3)',720,360,-180,-90,0.5);
write_ascii([pwd,'\',resultsfol,'\median_Magnitude_diff_counterfact_RP_histsoc-counterfact_EI_1901soc.asc'],nanmedian(Mcl_a30-Mcl_b30_19,3)',720,360,-180,-90,0.5);
write_ascii([pwd,'\',resultsfol,'\median_Intensit_diff_counterfact_RP_histsoc-counterfact_EI_1901soc.asc'],nanmedian(PIcl_a30-PIcl_b30_19,3)',720,360,-180,-90,0.5);
write_ascii([pwd,'\',resultsfol,'\median_DDD_diff_counterfact_RP_histsoc-counterfact_EI_1901soc.asc'],nanmedian(DDDcl_a30-DDDcl_b30_19,3)',720,360,-180,-90,0.5);
write_ascii([pwd,'\',resultsfol,'\median_DRD_diff_counterfact_RP_histsoc-counterfact_EI_1901soc.asc'],nanmedian(DRDcl_a30-DRDcl_b30_19,3)',720,360,-180,-90,0.5);

write_ascii([pwd,'\',resultsfol,'\median_Duration_diff_factual_RP_histsoc-counterfact_EI_1901c.asc'],nanmedian(Doc_a30-Dcl_b30_19,3)',720,360,-180,-90,0.5);
write_ascii([pwd,'\',resultsfol,'\median_Magnitude_diff_factual_RP_histsoc-counterfact_EI_1901.asc'],nanmedian(Moc_a30-Mcl_b30_19,3)',720,360,-180,-90,0.5);
write_ascii([pwd,'\',resultsfol,'\median_Intensit_diff_factual_RP_histsoc-counterfact_EI_1901.asc'],nanmedian(PIoc_a30-PIcl_b30_19,3)',720,360,-180,-90,0.5);
write_ascii([pwd,'\',resultsfol,'\median_DDD_diff_factual_RP_histsoc-counterfact_EI_1901.asc'],nanmedian(DDDoc_a30-DDDcl_b30_19,3)',720,360,-180,-90,0.5);
write_ascii([pwd,'\',resultsfol,'\median_DRD_diff_factual_RP_histsoc-counterfact_EI_1901.asc'],nanmedian(DRDoc_a30-DRDcl_b30_19,3)',720,360,-180,-90,0.5);
    
write_ascii([pwd,'\',resultsfol,'\median_Duration_counterfact_EI.asc'],nanmedian(Dcl_b30,3)',720,360,-180,-90,0.5);
write_ascii([pwd,'\',resultsfol,'\median_Magnitude_counterfact_EI.asc'],nanmedian(Mcl_b30,3)',720,360,-180,-90,0.5);
write_ascii([pwd,'\',resultsfol,'\median_Intensit_counterfacty_EI.asc'],nanmedian(PIcl_b30,3)',720,360,-180,-90,0.5);
write_ascii([pwd,'\',resultsfol,'\median_DDD_counterfact_EI.asc'],nanmedian(DDDcl_b30,3)',720,360,-180,-90,0.5);
write_ascii([pwd,'\',resultsfol,'\median_DRD_counterfact_EI.asc'],nanmedian(DRDcl_b30,3)',720,360,-180,-90,0.5);

write_ascii([pwd,'\',resultsfol,'\median_Duration_factual_EI.asc'],nanmedian(Doc_b30,3)',720,360,-180,-90,0.5);
write_ascii([pwd,'\',resultsfol,'\median_Magnitude_factual_EI.asc'],nanmedian(Moc_b30,3)',720,360,-180,-90,0.5);
write_ascii([pwd,'\',resultsfol,'\median_Intensity_factual_EI.asc'],nanmedian(PIoc_b30,3)',720,360,-180,-90,0.5);
write_ascii([pwd,'\',resultsfol,'\median_DDD_factual_EI.asc'],nanmedian(DDDoc_b30,3)',720,360,-180,-90,0.5);
write_ascii([pwd,'\',resultsfol,'\median_DRD_factual_EI.asc'],nanmedian(DRDoc_b30,3)',720,360,-180,-90,0.5);

write_ascii([pwd,'\',resultsfol,'\median_Duration_counterfact_RP.asc'],nanmedian(Dcl_a30,3)',720,360,-180,-90,0.5);
write_ascii([pwd,'\',resultsfol,'\median_Magnitude_counterfact_RP.asc'],nanmedian(Mcl_a30,3)',720,360,-180,-90,0.5);
write_ascii([pwd,'\',resultsfol,'\median_Intensity_counterfact_RP.asc'],nanmedian(PIcl_a30,3)',720,360,-180,-90,0.5);
write_ascii([pwd,'\',resultsfol,'\median_DDD_counterfact_RP.asc'],nanmedian(DDDcl_a30,3)',720,360,-180,-90,0.5);
write_ascii([pwd,'\',resultsfol,'\median_DRD_counterfact_RP.asc'],nanmedian(DRDcl_a30,3)',720,360,-180,-90,0.5);

write_ascii([pwd,'\',resultsfol,'\median_Duration_factual_RP.asc'],nanmedian(Doc_a30,3)',720,360,-180,-90,0.5);
write_ascii([pwd,'\',resultsfol,'\median_Magnitude_factual_RP.asc'],nanmedian(Moc_a30,3)',720,360,-180,-90,0.5);
write_ascii([pwd,'\',resultsfol,'\median_Intensity_factual_RP.asc'],nanmedian(PIoc_a30,3)',720,360,-180,-90,0.5);
write_ascii([pwd,'\',resultsfol,'\median_DDD_factual_RP.asc'],nanmedian(DDDoc_a30,3)',720,360,-180,-90,0.5);
write_ascii([pwd,'\',resultsfol,'\median_DRD_factual_RP.asc'],nanmedian(DRDoc_a30,3)',720,360,-180,-90,0.5);

write_ascii([pwd,'\',resultsfol,'\median_Duration_counterfact_EI_1901soc.asc'],nanmedian(Dcl_b30_19,3)',720,360,-180,-90,0.5);
write_ascii([pwd,'\',resultsfol,'\median_Magnitude_counterfact_EI_1901soc.asc'],nanmedian(Mcl_b30_19,3)',720,360,-180,-90,0.5);
write_ascii([pwd,'\',resultsfol,'\median_Intensity_counterfact_EI_1901soc.asc'],nanmedian(PIcl_b30_19,3)',720,360,-180,-90,0.5);
write_ascii([pwd,'\',resultsfol,'\median_DDD_counterfact_EI_1901soc.asc'],nanmedian(DDDcl_b30_19,3)',720,360,-180,-90,0.5);
write_ascii([pwd,'\',resultsfol,'\median_DRD_counterfact_EI_1901soc.asc'],nanmedian(DRDcl_b30_19,3)',720,360,-180,-90,0.5);

write_ascii([pwd,'\',resultsfol,'\median_Duration_factual_EI_1901soc.asc'],nanmedian(Doc_b30_19,3)',720,360,-180,-90,0.5);
write_ascii([pwd,'\',resultsfol,'\median_Magnitude_factual_EI_1901soc.asc'],nanmedian(Moc_b30_19,3)',720,360,-180,-90,0.5);
write_ascii([pwd,'\',resultsfol,'\median_Intensity_factual_EI_1901soc.asc'],nanmedian(PIoc_b30_19,3)',720,360,-180,-90,0.5);
write_ascii([pwd,'\',resultsfol,'\median_DDD_factual_EI_1901soc.asc'],nanmedian(DDDoc_b30_19,3)',720,360,-180,-90,0.5);
write_ascii([pwd,'\',resultsfol,'\median_DRD_factual_EI_1901soc.asc'],nanmedian(DRDoc_b30_19,3)',720,360,-180,-90,0.5);

write_ascii([pwd,'\',resultsfol,'\median_Duration_counterfact_RP_1901soc.asc'],nanmedian(Dcl_a30_19,3)',720,360,-180,-90,0.5);
write_ascii([pwd,'\',resultsfol,'\median_Magnitude_counterfact_RP_1901soc.asc'],nanmedian(Mcl_a30_19,3)',720,360,-180,-90,0.5);
write_ascii([pwd,'\',resultsfol,'\median_Intensity_counterfact_RP_1901soc.asc'],nanmedian(PIcl_a30_19,3)',720,360,-180,-90,0.5);
write_ascii([pwd,'\',resultsfol,'\median_DDD_counterfact_RP_1901soc.asc'],nanmedian(DDDcl_a30_19,3)',720,360,-180,-90,0.5);
write_ascii([pwd,'\',resultsfol,'\median_DRD_counterfact_RP_1901soc.asc'],nanmedian(DRDcl_a30_19,3)',720,360,-180,-90,0.5);

write_ascii([pwd,'\',resultsfol,'\median_Duration_factual_RP_1901soc.asc'],nanmedian(Doc_a30_19,3)',720,360,-180,-90,0.5);
write_ascii([pwd,'\',resultsfol,'\median_Magnitude_factual_RP_1901soc.asc'],nanmedian(Moc_a30_19,3)',720,360,-180,-90,0.5);
write_ascii([pwd,'\',resultsfol,'\median_Intensity_factual_RP_1901soc.asc'],nanmedian(PIoc_a30_19,3)',720,360,-180,-90,0.5);
write_ascii([pwd,'\',resultsfol,'\median_DDD_factual_RP_1901soc.asc'],nanmedian(DDDoc_a30_19,3)',720,360,-180,-90,0.5);
write_ascii([pwd,'\',resultsfol,'\median_DRD_factual_RP_1901soc.asc'],nanmedian(DRDoc_a30_19,3)',720,360,-180,-90,0.5);

save(resultsfol,'-v7.3');