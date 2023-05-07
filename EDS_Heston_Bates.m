tic;
% clc;
warning off;
format long;
S_0=100;%initial asset price

r=0.0367;q=0;lambda=6.21;v_bar=0.019;eta=0.61;rho=-0.7;v_0=0.101^2;%parameter of Heston model fusai 2016
K=[90;100;110];% strike price
Tmat=1;
N_monitor=12;
delta_t=Tmat/N_monitor;%maturity and the number of monitor dates


% r=0.0367;q=0;lambda=3.99;v_bar=0.014;eta=0.27;rho=-0.79;v_0=0.094^2;l_x=0.11;mu_x=-0.1391;sigma_x=0.15;%parameter of Bates model fusai 2016
% K=[90;100;110];% strike price
% Tmat=1;N_monitor=12;delta_t=Tmat/N_monitor;%maturity and the number of monitor dates


%% grid of state variable
L=86; 
% truncation range Heston model: L=86 for N=12, L=99 for N=25, L=96 for N=50; 
% Bates model: L=90 for N=12, L=92 for N=25, L=99 for N=50
n_r=620;n_l=80; 
% truncation level Heston model: n_r=620, n_l=80 for N=12, n_r=650, n_l=70 for N=25, n_r=1690, n_l=170 for N=50; 
% Bates model: n_r=580, n_l=60 for N=12, n_r=660, n_l=80 for N=25, n_r=1850, n_l=170 for N=50
delta_omega=L/n_r;
xi_m=-1.5;xi_p=-1.1;% damping factor
omega_m=(-n_l:n_r)'*delta_omega+1i*xi_m;omega_p=(-n_l:n_r)'*delta_omega+1i*xi_p;
omegam1=omega_m(n_l+1:end);
omegap1=omega_p(n_l+1:end);
%% grid of variance
q_CIR=2*lambda*v_bar/eta^2-1;
zeta_CIR2=2*lambda/((1-exp(-lambda*Tmat))*eta^2);   % This part is for determing the truancation boudary for log-variance. It will be used in conditional density p(sigma_T) condition on sigma_ini.
mean_var=log(v_0*exp(-lambda*Tmat)+v_bar*(1-exp(-lambda*Tmat)));
a_v_ini=mean_var-5/(1+q_CIR);    % a proper initial guess for interval boundary
b_v_ini=mean_var+2/(1+q_CIR);
sigma_ini=log(v_0);
TOL=10^(-7);                     % error tolerance for conditional density function
eps=10^(-6);                     % Choose a proper error tolerance for finding early exercise boundary and boundary for log variance
a_v=NewtonIterate_low(a_v_ini,zeta_CIR2,sigma_ini,lambda,Tmat,q_CIR,TOL,eps);    % interval boundary for log-variance. We use Tmat instead of Delta here.
b_v=NewtonIterate_up(b_v_ini,zeta_CIR2,sigma_ini,lambda,Tmat,q_CIR,TOL,eps);
n_v=39;                         % total nodes for log-variance. Heston model: J=39 for N=12, J=50 for N=25, J=68for N=50; Bates model: J=33 for N=12, J=46 for N=25, J=64 for N=50
low_var=exp(a_v);                % boundary for variance
up_var=exp(b_v);

[nodes,w]=lgwt(n_v,a_v,b_v);       % compute corresponding nodes and weights for Guass-Legendre.
Zeta=fliplr(nodes.');  

%% exponential factor
a=1.4;


%% preliminary compution

%% evaluation of Gamma function

gamma1m=2+1i*omega_m;gamma1p=2+1i*omega_p;
gamma2m1=1i*(omega_m-omega_p(1));gamma2m2=1i*(omega_m(1)-omega_p(2:end));
gamma2p1=1i*(omega_p-omega_m(1));gamma2p2=1i*(omega_p(1)-omega_m(2:end));

l=ceil(log2(2*(n_l+n_r+1)-1));
cm=zeros(2^l,1);cp=zeros(2^l,1);

%% Matlab
gamma1m1=cgamma(gamma1m(n_l+1:end));
gamma1p1=cgamma(gamma1p(n_l+1:end));
Gamma1m=exp(a*real(omega_m)).*[flipud(conj(gamma1m1(2:n_l+1)));gamma1m1];
Gamma1p=exp(a*real(omega_p)).*[flipud(conj(gamma1p1(2:n_l+1)));gamma1p1];
 
gamma2m11=cgamma(gamma2m1);
gamma2m21=conj(gamma2m11(2:end));
Gamma2m1=exp(a*real(gamma2m1/1i)).*gamma2m11;
Gamma2m2=exp(a*real(gamma2m2/1i)).*gamma2m21;

gamma2p11=cgamma(gamma2p1);
gamma2p21=conj(gamma2p11(2:end));
Gamma2p1=exp(a*real(gamma2p1/1i)).*gamma2p11;
Gamma2p2=exp(a*real(gamma2p2/1i)).*gamma2p21;



Gammainvm=1./Gamma1m(n_l+1:end);
Gammainvp=1./Gamma1p(n_l+1:end);

cm(1:n_l+n_r+1)=Gamma2m1;
cm(2^l-(n_l+n_r-1):2^l)=flipud(Gamma2m2);
cp(1:n_l+n_r+1)=Gamma2p1;
cp(2^l-(n_l+n_r-1):2^l)=flipud(Gamma2p2);




%% product of conditional ChF of increments and transition density of CIR
%% Heston & Bates
Chara_m=zeros(n_l+n_r+1,n_v,n_v);
Chara_p=zeros(n_l+n_r+1,n_v,n_v);

k_i=0;k_m=0;k_p=0;% Heston 
% k_i=l_x*(exp(mu_x+sigma_x^2/2)-1);% Bates
% k_m=l_x*(exp(-1i*omegam1*mu_x-sigma_x^2*omegam1.^2/2)-1)*delta_t;
% k_p=l_x*(exp(-1i*omegap1*mu_x-sigma_x^2*omegap1.^2/2)-1)*delta_t;

Zeta_t=reshape(Zeta,1,[],1);
Zeta_s=reshape(Zeta,1,1,[]);
psi_m=sqrt(lambda^2+2*1i*(omegam1.*(rho*lambda/eta-1/2-1i*omegam1*(1-rho^2)/2))*eta^2);
psi_p=sqrt(lambda^2+2*1i*(omegap1.*(rho*lambda/eta-1/2-1i*omegap1*(1-rho^2)/2))*eta^2);
Bessel_chf_m_1=4*psi_m.*exp(-1/2*psi_m*delta_t)./(eta^2*(1-exp(-psi_m*delta_t)));
Bessel_chf_p_1=4*psi_p.*exp(-1/2*psi_p*delta_t)./(eta^2*(1-exp(-psi_p*delta_t)));
Bessel_chf_2=exp((Zeta_t+Zeta_s)/2);
Chara_m_1=2*psi_m.*exp(-1/2*(psi_m-lambda)*delta_t)./(eta^2*(1-exp(-psi_m*delta_t)));
Chara_p_1=2*psi_p.*exp(-1/2*(psi_p-lambda)*delta_t)./(eta^2*(1-exp(-psi_p*delta_t)));
Chara_2=exp(Zeta_t).*exp(q_CIR/2*(Zeta_t-(Zeta_s-lambda*delta_t)));
Exp1m=-1i*omegam1*(r-q-k_i-rho*lambda*v_bar/eta)*delta_t+k_m;
Exp1p=-1i*omegap1*(r-q-k_i-rho*lambda*v_bar/eta)*delta_t+k_p;
Exp2m=(1i*omegam1*rho*eta+lambda)/eta^2;
Exp2p=(1i*omegap1*rho*eta+lambda)/eta^2;
Exp3m=psi_m.*(1+exp(-psi_m*delta_t))./((1-exp(-psi_m*delta_t))*eta^2);
Exp3p=psi_p.*(1+exp(-psi_p*delta_t))./((1-exp(-psi_p*delta_t))*eta^2);
Zeta_m=exp(Zeta_t)-exp(Zeta_s);
Zeta_p=exp(Zeta_t)+exp(Zeta_s);

Bessel_chf_m=Bessel_chf_m_1.*Bessel_chf_2;
Bessel_chf_p=Bessel_chf_p_1.*Bessel_chf_2;
Charam1=Chara_m_1.*Chara_2.*exp(Exp1m-Exp2m.*Zeta_m-Exp3m.*Zeta_p+abs(real((Bessel_chf_m)))).*(besseli(q_CIR,Bessel_chf_m,1));
Chara_m(n_l+1:end,:,:)=Charam1;
Chara_m(1:n_l,:,:)=conj(Charam1(n_l+1:-1:2,:,:));
Charap1=Chara_p_1.*Chara_2.*exp(Exp1p-Exp2p.*Zeta_m-Exp3p.*Zeta_p+abs(real((Bessel_chf_p)))).*(besseli(q_CIR,Bessel_chf_p,1));
Chara_p(n_l+1:end,:,:)=Charap1;
Chara_p(1:n_l,:,:)=conj(Charap1(n_l+1:-1:2,:,:));

Bessel_chf_2_0=exp(Zeta/2)*sqrt(v_0);
Chara_2_0=exp(Zeta).*exp(q_CIR/2*(Zeta-(log(v_0)-lambda*delta_t)));

Bessel_chf_m_0=Bessel_chf_m_1*Bessel_chf_2_0;
Charavm01=(Chara_m_1*Chara_2_0.*exp(Exp1m-Exp2m*(exp(Zeta)-v_0)-Exp3m*(exp(Zeta)+v_0)+abs(real(Bessel_chf_m_0))).*besseli(q_CIR,Bessel_chf_m_0,1));

Bessel_chf_p_0=Bessel_chf_p_1*Bessel_chf_2_0;
Charavp01=(Chara_p_1*Chara_2_0.*exp(Exp1p-Exp2p*(exp(Zeta)-v_0)-Exp3p*(exp(Zeta)+v_0)+abs(real(Bessel_chf_p_0))).*besseli(q_CIR,Bessel_chf_p_0,1));


%% backward induction

W_hat_m_1=Gamma1m./(omega_m.*(1i-omega_m));
W_hat_p_1=Gamma1p./(omega_p.*(1i-omega_p));

W_hat_m=sum(W_hat_m_1.*Chara_m.*w.',2);
W_hat_p=sum(W_hat_p_1.*Chara_p.*w.',2);
W_hat_m=reshape(W_hat_m,n_l+n_r+1,n_v,1);
W_hat_p=reshape(W_hat_p,n_l+n_r+1,n_v,1);


 
for t_k=1:N_monitor-2
    W_m=W_hat_m;
    W_p=W_hat_p;
    T_fft_m=ifft(fft(cm).*fft([W_p;zeros(2^l-(n_l+n_r+1),n_v)]));
    T_fft_p=ifft(fft(cp).*fft([W_m;zeros(2^l-(n_l+n_r+1),n_v)]));
    T_m=T_fft_m(1:n_l+n_r+1,:);
    T_p=T_fft_p(1:n_l+n_r+1,:);
    W_hat_m=1/(2*pi)*sum(Chara_m.*T_m.*w.',2)*delta_omega;
    W_hat_p=sum(Chara_p.*w.'.*(W_p+1/(2*pi)*T_p*delta_omega),2);
    W_hat_m=reshape(W_hat_m,n_l+n_r+1,n_v,1);
    W_hat_p=reshape(W_hat_p,n_l+n_r+1,n_v,1);
end

W_m=W_hat_m;
W_p=W_hat_p;
T_fft_m=ifft(fft(cm).*fft([W_p;zeros(2^l-(n_l+n_r+1),n_v)]));
T_fft_p=ifft(fft(cp).*fft([W_m;zeros(2^l-(n_l+n_r+1),n_v)]));
T_m=T_fft_m(1:n_l+n_r+1,:);
T_p=T_fft_p(1:n_l+n_r+1,:);


x_0=-log((N_monitor+1)*K./S_0-1);

ww=[1/2,ones(1,n_r)];

%% Fourier inversion
W_hat_m_v0=1/(2*pi)*(Charavm01.*T_m(n_l+1:end,:))*delta_omega*w;
V_p_m=exp(-r*Tmat)/(N_monitor+1)/pi*real(max((N_monitor+1)*K-S_0,0).*exp(-1i*x_0*omegam1.').*ww*(W_hat_m_v0.*Gammainvm)*delta_omega);
V_c_m=(V_p_m+exp(-r*Tmat)*(S_0/(N_monitor+1)*(1-exp((r-q)*(N_monitor+1)*delta_t))/(1-exp((r-q)*delta_t))-K));


W_hat_p_v0=(Charavp01.*(W_p(n_l+1:end,:)+1/(2*pi)*T_p(n_l+1:end,:)*delta_omega))*w;
V_p_p=exp(-r*Tmat)/(N_monitor+1)/pi*real(max((N_monitor+1)*K-S_0,0).*exp(-1i*x_0*omegap1.').*ww*(W_hat_p_v0.*Gammainvp)*delta_omega);
V_c_p=(V_p_p+exp(-r*Tmat)*(S_0/(N_monitor+1)*(1-exp((r-q)*(N_monitor+1)*delta_t))/(1-exp((r-q)*delta_t))-K));


%% Greeks
% Delta_p_m=-V_p_m./((N_monitor+1).*K-S_0)-exp(-r*Tmat)*K/(pi*S_0).*real(1i*omegam1.'.*exp(-1i*x_0*omegam1.').*ww*(W_hat_m_v0.*Gammainvm)*delta_omega);
% Delta_c_m=Delta_p_m+exp(-r*Tmat)/(N_monitor+1)*(1-exp((r-q)*(N_monitor+1)*delta_t))/(1-exp((r-q)*delta_t));
% Gamma_p_m=exp(-r*Tmat)*(N_monitor+1)*K.^2./(pi*S_0^2.*((N_monitor+1)*K-S_0)).*real(1i*omegam1.'.*(1+1i*omegam1.').*exp(-1i*x_0*omegam1.').*ww*(W_hat_m_v0.*Gammainvm)*delta_omega);
% Gamma_c_m=Gamma_p_m;

%% Richardson extrapolation
% V_c_250=((72*V_c_12-475*V_c_25+650*V_c_50)/247*18-2*V_c_25+9*V_c_50)/25;% V_c_N, N=12, 25, 50 represent option prices with the number of monitoring points N


toc;