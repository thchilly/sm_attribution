% Standardized Drought Analysis Toolbox (SDAT)
% - SPI: Standardized Precipitation Index
% - SSI: Standardized Soil Moisture Index
% - SRI: Standardized Runoff Index
% - SRHI: Standardized Relative Humidity Index
% - SGI: Standardised Groundwater level Index
% - Standardized Surface Water Supply Index (SSWSI)
% - Standardized Water Storage Index (SWSI)
% Input data should be a vector of precipitation, soil moisture, etc.
% sc: scale of the index
% Release 02/01/2015
%Refrences:
%Farahmand A., AghaKouchak A., 2015, A Generalized Framework for Deriving Nonparametric Standardized Drought Indicators, Advances in Water Resources, 76, 140-145, doi: 10.1016/j.advwatres.2014.11.012
%download reference:  http://amir.eng.uci.edu/publications/15_Drought_Standardized_Index_AWR.pdf
%Hao Z., AghaKouchak A., Nakhjiri N., Farahmand A., 2014, Global Integrated Drought Monitoring and Prediction System, Scientific Data, 1:140001, 1-10, doi: 10.1038/sdata.2014.1.
%download reference:  http://www.nature.com/articles/sdata20141
% Please read the disclaimer before using SDAT (Disclaimer.txt). By using SDAT users agree with the disclaimer.
%% main code
function [SI_ref, SI_proj]=SSI_mod(td_ref,td_proj,sc)
% sample input
% Sample input data (e.g., a vector of precipitation data)
% load precip.txt
% td=precip;
% sc: scale of the index (>1, e.g., 3-month SPI or SSI)
% sc=6;
%%moda
td_ref=[td_ref; td_proj];
%%
n_ref=length(td_ref);
n_proj=length(td_proj);
SI_ref=zeros(n_ref,1);
SI_proj=zeros(n_proj,1);
% Compute the SPI for each grid from the prcp or smc data
%For some grid, no observation exist.
if length(td_ref(td_ref>=0))/length(td_ref)~=1
    SI_ref(n_ref,1)=nan;
    SI_proj(n_proj,1)=nan;
else
    % Obtain the prcp and smc for the specified time scale and
    % compute the standarized drought index (for SPI and SSI)
    SI_ref(1:sc-1,1)=nan;
    SI_proj(1:sc-1,1)=nan;
    
    A1_ref=[];
    A1_proj=[];
    for i=1:sc
        A1_ref=[A1_ref,td_ref(i:length(td_ref)-sc+i)];
        A1_proj=[A1_proj,td_proj(i:length(td_proj)-sc+i)];
    end
    Y_ref=sum(A1_ref,2);
    Y_proj=sum(A1_proj,2);
    % Compute the SPI or SSI
    nn_ref=length(Y_ref);
    nn_proj=length(Y_proj);
    SI1p_ref=zeros(nn_ref,1);
    SI1p_proj=zeros(nn_proj,1);
    SI1_ref=zeros(nn_ref,1);
    SI1_proj=zeros(nn_proj,1);
    for k=1:12
        d_ref=Y_ref(k:12:nn_ref);
        d_proj=Y_proj(k:12:nn_proj);
        %compute the empirical probability
        nnn_ref=length(d_ref);
        nnn_proj=length(d_proj);
        bp_ref=zeros(nnn_ref,1);
        bp_proj=zeros(nnn_proj,1);
        for i=1:nnn_ref
            bp_ref(i,1)=sum(d_ref(:,1)<=d_ref(i,1));
        end
        for i=1:nnn_proj
            bp_proj(i,1)=sum(d_ref(:,1)<=d_proj(i,1));
        end
        y_ref=(bp_ref-0.44)./(nnn_ref+0.12);
        y_proj=(bp_proj-0.44)./(nnn_ref+0.12);
        SI1p_ref(k:12:nn_ref,1)=y_ref;
        SI1p_proj(k:12:nn_proj,1)=y_proj;
    end

    % % t1=real(sqrt(log(1./SI1p_ref.^2)));
    % % t2=real(sqrt(log(1/(1-SI1p_ref.^2))));
    % % SI1_ref1=-t1-((2.515517+0.802583.*t1+0.010328.*t1.^2)./(1+0.010328.*t1+0.010328.*t1.^2+0.189269.*t1.^3));
    % % SI1_ref2=-t2-((2.515517+0.802583.*t2+0.010328.*t2.^2)./(1+0.010328.*t2+0.010328.*t2.^2+0.189269.*t2.^3));
    % % SI1_ref(SI1p_ref<=0.5)=SI1_ref1(SI1p_ref<=0.5);
    % % SI1_ref(SI1p_ref>0.5)=SI1_ref2(SI1p_ref>0.5);

    SI1_ref=norminv(SI1p_ref(:,1));
    SI1_proj(:,1)=norminv(SI1p_proj(:,1));
    %output
    SI_ref(sc:end,1)=SI1_ref;
    SI_proj(sc:end,1)=SI1_proj;
    if isnan(sum(SI_proj(3:end)))
        stamata
    end
end
% plot(SI1_ref); hold on; plot(SI_proj);
 