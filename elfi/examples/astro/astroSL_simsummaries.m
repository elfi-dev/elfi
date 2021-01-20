function sim_summ = astroSL_simsummaries(bigtheta,n,nbin,numsim)

% simunlate n x numsim "data points" iid from a Gaussian with mean 0.5 and standard deviation 0.05,
% truncated to the interval [0.01,1.2].
%zdata = mytruncgaussrandraw_for_correlated_SL([0.01,1.2,0.5,0.05],uniform_draws);
pd = makedist('Normal', 0.5, 0.05);
pd = truncate(pd,0.01,1.2);
zdata = random(pd, n, numsim);


logom = bigtheta(1);
ok = bigtheta(2);
w0 = bigtheta(3);
wa = bigtheta(4);
logh0 = bigtheta(5);

om = exp(logom);
ol = 1-om;
h0 = exp(logh0);


zcentre = zeros(nbin,numsim);
for jj=1:numsim
   [~,zcentre(:,jj)]= hist(zdata(:,jj),nbin);  % bin the data and collect the centres of the bins
end

sim_summ = zeros(nbin,numsim);       % this will collect the simulated "summaries"


for jj=1:numsim
    for ii=1:nbin
        sim_summ(ii,jj) = distmodulus(zcentre(ii,jj),om,ok,ol,w0,wa,h0);
    end
end

end  

% end of main function astroSL_model

function mu = distmodulus(z,om,ok,ol,w0,wa,h0)
   mu = 5*log10(d_l(z,om,ok,ol,w0,wa,h0)*1e6/10); 
end

function luminosity = d_l(z,om,ok,ol,w0,wa,h0)
    luminosity = (1+z).*d_m(z,om,ok,ol,w0,wa)/h0;
end

function comdist = d_m(z,om,ok,ol,w0,wa)
  % returns comoving distance in Mpc/h '''
   comdist = d_c(z,om,ok,ol,w0,wa);
end

function out = d_c(z_integr_upper,om,ok,ol,w0,wa)
    c_km_per_s = 299792.458;  % speed of light km/s
    d_h = c_km_per_s/(100); % Mpc/h
  %  out =  d_h*integral(@(z)e_z_inverse(z,om,ok,ol,w0,wa),0,z_integr_upper,'ArrayValued',true)
    e_z_inverse = @(z) 1./(sqrt(om*(1+z).^3 + ok*(1+z).^2 + ol*exp(w_int(z,w0,wa))));
    out =  d_h*quadv(e_z_inverse,0,z_integr_upper);
end

%function out = e_z_inverse(z,om,ok,ol,w0,wa)
%    out = 1./(sqrt(om*(1+z).^3 + ok*(1+z).^2 + ol*exp(w_int(z,w0,wa))));
%end

function out = w_int(z,w0,wa)
        a=1./(1+z);
        out = 3*quadv(@(avar)w_integrand(avar,w0,wa),a,1);
end

% function out = w_int(z,w0,wa)
%         a=1/(1+z);
%         out = zeros(length(a),1);
%          for ii=1:length(a)
%            out(ii) = 3.0*integral(@(a)w_integrand(a(ii),w0,wa),a(ii),1.0,'ArrayValued',true,'RelTol',1e-6);
%          end
% end

function out = w_integrand(a,w0,wa)
     out = (1+ wfunc(a,w0,wa))./a;
end

function out = wfunc(a,w0,wa) % e.g. Linder 2003
  out = w0 + (1-a)*wa;
end


