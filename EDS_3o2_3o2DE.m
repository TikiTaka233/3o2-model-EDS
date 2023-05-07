tic;
% clc;
warning off;
format long;
S_0=100;%initial asset price

r=0.05;q=0;lambda=22.84;v_bar=0.4669^2;eta=8.56;rho=-0.99;v_0=0.060025;%parameter of 3/2 model 
K=[90;100;110];% strike price
Tmat=1;N_monitor=12;delta_t=Tmat/N_monitor;%maturity and the number of monitor dates


% r=0.05;q=0;lambda=22.84;v_bar=0.4669^2;eta=8.56;rho=-0.99;v_0=0.060025;l_x=5;p=0.4;eta_x1=10;eta_x2=5;%parameter of 3/2 model with DE jump
% K=[90;100;110];% strike price
% Tmat=1;N_monitor=12;delta_t=Tmat/N_monitor;%maturity and the number of monitor dates

%% grid of state variable
L=37; 
% truncation range 3/2 model: L=37 for N=12, L=43 for N=25, L=41 for N=50; 
% 3/2+DE model: L=23 for N=12, L=30 for N=25, L=30 for N=50
n_r=240;n_l=40; 
% truncation level 3/2 model: n_r=240, n_l=40 for N=12, n_r=320, n_l=50 for N=25, n_r=470, n_l=130 for N=50; 
% 3/2+DE model: n_r=150, n_l=40 for N=12, n_r=190, n_l=30 for N=25, n_r=270, n_l=80 for N=50
delta_omega=L/n_r;
xi_m=-1.5;xi_p=-1.1;% damping factor
omega_m=(-n_l:n_r)'*delta_omega+1i*xi_m;omega_p=(-n_l:n_r)'*delta_omega+1i*xi_p;
omegam1=omega_m(n_l+1:end);
omegap1=omega_p(n_l+1:end); 

%% grid of recipal of variance
lambda1=lambda*v_bar;v_bar1=(lambda+eta^2)/(lambda*v_bar);eta1=-eta;
v_ini=1/v_0;
q_CIR=2*lambda1*v_bar1/eta1^2-1;
zeta_CIR=2*lambda1/((1-exp(-lambda1*delta_t))*eta1^2);   % zeta_CIR is large, as lambda*Delta is close to zero
zeta_CIR2=2*lambda1/((1-exp(-lambda1*Tmat))*eta1^2);
mean_var=log(v_ini*exp(-lambda1*Tmat)+v_bar1*(1-exp(-lambda1*Tmat)));
a_v_ini=mean_var-5/(1+q_CIR);    % a proper initial guess for interval boundary
b_v_ini=mean_var+2/(1+q_CIR);
sigma_ini=log(v_ini);
TOL=10^(-7);                     % error tolerance for conditional density function
eps=10^(-5);                     % Choose a proper error tolerance for finding early exercise boundary and boundary for log variance
a_v=NewtonIterate_low(a_v_ini,zeta_CIR2,sigma_ini,lambda1,Tmat,q_CIR,TOL,eps);    % interval boundary for log-variance. We use Tmat instead of Delta here.
b_v=NewtonIterate_up(b_v_ini,zeta_CIR2,sigma_ini,lambda1,Tmat,q_CIR,TOL,eps);
n_v=27;                         % total nodes for log-variance. 3/2 model: J=27 for N=12, J=32 for N=25, J=49 for N=50; 3/2+DE model: J=26 for N=12, J=37 for N=25, J=51 for N=50
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
%% 3/2 model & 3/2+DE model
v1=q_CIR;zeta1=zeta_CIR;
Chara_m=zeros(n_l+n_r+1,n_v,n_v);
Chara_p=zeros(n_l+n_r+1,n_v,n_v);

k_i=0;k_m=0;k_p=0; % no jumps
% k_i=l_x*(p*eta_x1/(eta_x1-1)+(1-p)*eta_x2/(eta_x2+1)-1); % DE jumps
% k_m=l_x*(p*eta_x1./(eta_x1+1i*omegam1)+(1-p)*eta_x2./(eta_x2-1i*omegam1)-1)*delta_t;
% k_p=l_x*(p*eta_x1./(eta_x1+1i*omegap1)+(1-p)*eta_x2./(eta_x2-1i*omegap1)-1)*delta_t;

nu_m=sqrt(v1^2+8i*omegam1*(rho*lambda/eta+rho*eta/2-1/2)/eta^2+4*omegam1.^2*(1-rho^2)/eta^2);
nu_p=sqrt(v1^2+8i*omegap1*(rho*lambda/eta+rho*eta/2-1/2)/eta^2+4*omegap1.^2*(1-rho^2)/eta^2);

Bessel_var=2*zeta1*exp(-1/2*lambda*v_bar*delta_t+1/2*(Zeta'+Zeta));
Bessel_var_upper=triu(Bessel_var);
w_diag=ones(n_r+1,n_v,n_v)+reshape(-0.5*eye(n_v,n_v),1,n_v,n_v);

Bessel_threshold=80;
Index_Besselvarupper1=find(Bessel_var_upper>=Bessel_threshold);
Index_Besselvarupper2=find(Bessel_var_upper~=0&Bessel_var_upper<Bessel_threshold);
Besselvarupper_Matrix=zeros(n_r+1,1,1)+reshape(Bessel_var_upper,1,n_v,n_v);
Index_Besselvarupper_Matrix1=find(Besselvarupper_Matrix>=Bessel_threshold);
Index_Besselvarupper_Matrix2=find(Besselvarupper_Matrix~=0&Besselvarupper_Matrix<Bessel_threshold);
Bessel_chf_m=zeros(n_r+1,n_v,n_v);Bessel_chf_p=zeros(n_r+1,n_v,n_v);
Bessel_chf_m(Index_Besselvarupper_Matrix2)=reshape(cbesseli(nu_m,Bessel_var(Index_Besselvarupper2)),[],1);
Bessel_chf_p(Index_Besselvarupper_Matrix2)=reshape(cbesseli(nu_p,Bessel_var(Index_Besselvarupper2)),[],1);
math('matlab2math','Num',nu_m+zeros(1,length(Index_Besselvarupper1)));
math('matlab2math','Nup',nu_p+zeros(1,length(Index_Besselvarupper1)));
math('matlab2math','Besselvar',Bessel_var_upper(Index_Besselvarupper1).'+zeros(n_r+1,1));
Bessel_chf_m(Index_Besselvarupper_Matrix1)=math('math2matlab','Exp[-Besselvar]*BesselI[Num,Besselvar]+0.I');
Bessel_chf_p(Index_Besselvarupper_Matrix1)=math('math2matlab','Exp[-Besselvar]*BesselI[Nup,Besselvar]+0.I');
Bessel_chf_m=(Bessel_chf_m+permute(Bessel_chf_m,[1,3,2])).*w_diag;
Bessel_chf_p=(Bessel_chf_p+permute(Bessel_chf_p,[1,3,2])).*w_diag;

Chara_m_1=zeta1*exp(-1i*omegam1*(r-q-k_i-rho*lambda*v_bar/eta)*delta_t+Zeta+v1*lambda*v_bar/2*delta_t+k_m);
Chara_p_1=zeta1*exp(-1i*omegap1*(r-q-k_i-rho*lambda*v_bar/eta)*delta_t+Zeta+v1*lambda*v_bar/2*delta_t+k_p);
Zeta_m=Zeta'-Zeta;
Zeta_m=reshape(Zeta_m,1,n_v,n_v);
Exp1m=1i*omegam1*rho/eta+v1/2;
Exp1p=1i*omegap1*rho/eta+v1/2;
Exp2=-zeta1*(exp(Zeta')+exp(Zeta-lambda*v_bar*delta_t));
Exp2=reshape(Exp2,1,n_v,n_v);
Bessel_var=reshape(Bessel_var,1,n_v,n_v);
Charam1=Chara_m_1.*exp(Exp1m.*Zeta_m+Exp2+Bessel_var).*Bessel_chf_m;
Charap1=Chara_p_1.*exp(Exp1p.*Zeta_m+Exp2+Bessel_var).*Bessel_chf_p;
Chara_m(n_l+1:end,:,:)=Charam1;Chara_m(1:n_l,:,:)=conj(Charam1(n_l+1:-1:2,:,:));
Chara_p(n_l+1:end,:,:)=Charap1;Chara_p(1:n_l,:,:)=conj(Charap1(n_l+1:-1:2,:,:));

Bessel_var_0=2*zeta1*exp(-1/2*lambda*v_bar*delta_t+1/2*(Zeta-log(v_0)));
Index_Besselvar01=find(Bessel_var_0>=Bessel_threshold);
Index_Besselvar02=find(Bessel_var_0~=0&Bessel_var_0<Bessel_threshold);

Bessel_chf_m=zeros(n_r+1,n_v);
Bessel_chf_m(:,Index_Besselvar02)=cbesseli(nu_m,Bessel_var_0(Index_Besselvar02));
math('matlab2math','Num0',nu_m+zeros(1,length(Index_Besselvar01)));
math('matlab2math','Besselvar0',Bessel_var_0(Index_Besselvar01)+zeros(n_grid+1,1));
Bessel_chf_m(:,Index_Besselvar01)=math('math2matlab','Exp[-Besselvar0]*BesselI[Num0,Besselvar0]');
Exp1_m_0=(1i*omegam1*rho/eta+v1/2)*(Zeta+log(v_0));
Exp2_0=-zeta1*(exp(Zeta)+exp(-lambda*v_bar*delta_t)/v_0);
Charavm01=(Chara_m_1.*exp(Exp1_m_0+Exp2_0+1*Bessel_var_0).*Bessel_chf_m);

Bessel_chf_p=zeros(n_r+1,n_v);
Bessel_chf_p(:,Index_Besselvar02)=cbesseli(nu_p,Bessel_var_0(Index_Besselvar02));
math('matlab2math','Nup0',nu_p+zeros(1,length(Index_Besselvar01)));
Bessel_chf_p(:,Index_Besselvar01)=math('math2matlab','Exp[-Besselvar0]*BesselI[Nup0,Besselvar0]');
Exp1_p_0=(1i*omegap1*rho/eta+v1/2)*(Zeta+log(v_0));
Charavp01=(Chara_p_1.*exp(Exp1_p_0+Exp2_0+1*Bessel_var_0).*Bessel_chf_p);

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