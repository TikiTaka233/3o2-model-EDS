tic;
% clc;
warning off;
format long;
S_0=[92;96;100;104;108];%initial asset price

r=0.05;q=0;sigma1=0.15;sigma2=0.25;a1=-0.10;a2=-0.10;b1=0.30;b2=0.30;lambda1=5;lambda2=2;% J.L.Kirkby && regime switching Merton-Merton
Q=[-0.5,0.5;0.5,-0.5];% CTMC generator 
K=100;% strike price
Tmat=1;
N_monitor=12;
delta_t=Tmat/N_monitor;

% r=0.05;q=0;sigma1=0.15;sigma2=0.25;a1=0.3753;a2=-0.5503;b1=0.18;b2=0.6944;eta1=3.0465;eta2=3.0775;p1=0.3445;p2=0.3445;lambda1=5;lambda2=2;% J.L.Kirkby && regime switching && Kou-MN(mixed normal)
% Q=[-0.5,0.5;0.5,-0.5];% CTMC generator 
% K=100;% strike price
% Tmat=1;N_monitor=12;delta_t=Tmat/N_monitor;

%% grid of state variable
L=25;
% truncation range Merton-Merton: L=25 for N=12, L=30 for N=25, L=30 for N=50; 
% Kou-MN: L=28 for N=12, L=30 for N=25, L=35 for N=50
n_l=180;n_r=40;
% truncation level Merton-Merton: n_r=180, n_l=40 for N=12, n_r=230, n_l=60 for N=25, n_r=290, n_l=100 for N=50; 
% Kou-MN: n_r=190, n_l=40 for N=12, n_r=210, n_l=50 for N=25, n_r=380, n_l=110 for N=50
delta_omega=L/n_r;
xi_m=-1.5;xi_p=-1.1;% dampening factor
omega_m=(-n_l:n_r)'*delta_omega+1i*xi_m;omega_p=(-n_l:n_r)'*delta_omega+1i*xi_p;
omegam1=omega_m(n_l+1:end);
omegap1=omega_p(n_l+1:end);


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

%% regime switching
%% Merton+Merton
mu1=r-q-sigma1^2/2-lambda1*(exp(a1+1/2*b1^2)-1);
mu2=r-q-sigma2^2/2-lambda2*(exp(a2+1/2*b2^2)-1);
Psi_m1=-1i*omegam1*mu1-1/2*omegam1.^2*sigma1^2+lambda1*(exp(-1i*omegam1*a1-1/2*omegam1.^2*b1^2)-1);% regime 1
Psi_p1=-1i*omegap1*mu1-1/2*omegap1.^2*sigma1^2+lambda1*(exp(-1i*omegap1*a1-1/2*omegap1.^2*b1^2)-1);
Psi_m2=-1i*omegam1*mu2-1/2*omegam1.^2*sigma2^2+lambda2*(exp(-1i*omegam1*a2-1/2*omegam1.^2*b2^2)-1);% regime 2
Psi_p2=-1i*omegap1*mu2-1/2*omegap1.^2*sigma2^2+lambda2*(exp(-1i*omegap1*a2-1/2*omegap1.^2*b2^2)-1);
Epsilon_diag_m=zeros(2,2,n_r+1);Epsilon_diag_p=zeros(2,2,n_r+1);
Epsilon_diag_m(1,1,:)=Psi_m1;Epsilon_diag_m(2,2,:)=Psi_m2;
Epsilon_diag_p(1,1,:)=Psi_p1;Epsilon_diag_p(2,2,:)=Psi_p2;

Epsilon_diag_m=Epsilon_diag_m+Q';Epsilon_diag_p=Epsilon_diag_p+Q';
Eig_m_1=1/2*(Epsilon_diag_m(1,1,:)+Epsilon_diag_m(2,2,:)-sqrt(Epsilon_diag_m(1,1,:).^2+Epsilon_diag_m(2,2,:).^2-...
    2*Epsilon_diag_m(1,1,:).*Epsilon_diag_m(2,2,:)+4*Epsilon_diag_m(1,2,:).*Epsilon_diag_m(2,1,:)));
Eig_m_2=1/2*(Epsilon_diag_m(1,1,:)+Epsilon_diag_m(2,2,:)+sqrt(Epsilon_diag_m(1,1,:).^2+Epsilon_diag_m(2,2,:).^2-...
    2*Epsilon_diag_m(1,1,:).*Epsilon_diag_m(2,2,:)+4*Epsilon_diag_m(1,2,:).*Epsilon_diag_m(2,1,:)));
Eig_p_1=1/2*(Epsilon_diag_p(1,1,:)+Epsilon_diag_p(2,2,:)-sqrt(Epsilon_diag_p(1,1,:).^2+Epsilon_diag_p(2,2,:).^2-...
    2*Epsilon_diag_p(1,1,:).*Epsilon_diag_p(2,2,:)+4*Epsilon_diag_p(1,2,:).*Epsilon_diag_p(2,1,:)));
Eig_p_2=1/2*(Epsilon_diag_p(1,1,:)+Epsilon_diag_p(2,2,:)+sqrt(Epsilon_diag_p(1,1,:).^2+Epsilon_diag_p(2,2,:).^2-...
    2*Epsilon_diag_p(1,1,:).*Epsilon_diag_p(2,2,:)+4*Epsilon_diag_p(1,2,:).*Epsilon_diag_p(2,1,:)));
Epsilon_m=exp(Eig_m_1*delta_t).*(eye(2,2)+zeros(2,2,n_grid+1))+(exp(Eig_m_1*delta_t)-exp(Eig_m_2*delta_t))./(Eig_m_1-Eig_m_2).*(Epsilon_diag_m-Eig_m_1.*(eye(2,2)+zeros(2,2,n_grid+1)));
Epsilon_p=exp(Eig_p_1*delta_t).*(eye(2,2)+zeros(2,2,n_grid+1))+(exp(Eig_p_1*delta_t)-exp(Eig_p_2*delta_t))./(Eig_p_1-Eig_p_2).*(Epsilon_diag_p-Eig_p_1.*(eye(2,2)+zeros(2,2,n_grid+1)));

n_v=2;
Chara_m=zeros(n_l+n_r+1,n_v,n_v);
Chara_p=zeros(n_l+n_r+1,n_v,n_v);
Chara_m(n_l+1:end,:,:)=permute(Epsilon_m,[3,2,1]);
Chara_m(1:n_l,:,:)=conj(Chara_m(2*n_l+1:-1:n_l+2,:,:));
Chara_p(n_l+1:end,:,:)=permute(Epsilon_p,[3,2,1]);
Chara_p(1:n_l,:,:)=conj(Chara_p(2*n_l+1:-1:n_l+2,:,:));
w=ones(n_v,1);

%% Kou+MN
% mu1=r-q-sigma1^2/2-lambda1*(p1*eta1/(eta1-1)+(1-p1)*eta2/(eta2+1)-1);% double exponential
% mu2=r-q-sigma2^2/2-lambda2*(p2*exp(a1+1/2*b1^2)+(1-p2)*exp(a2+1/2*b2^2)-1);% mixed normal
% Psi_m1=-1i*omegam1*mu1-1/2*omegam1.^2*sigma1^2+lambda1*(p1*eta1./(eta1+1i*omegam1)+(1-p1)*eta2./(eta2-1i*omegam1)-1);% regime 1
% Psi_p1=-1i*omegap1*mu1-1/2*omegap1.^2*sigma1^2+lambda1*(p1*eta1./(eta1+1i*omegap1)+(1-p1)*eta2./(eta2-1i*omegap1)-1);
% Psi_m2=-1i*omegam1*mu2-1/2*omegam1.^2*sigma2^2+lambda2*(p2*exp(-1i*omegam1*a1-1/2*omegam1.^2*b1^2)+(1-p2)*exp(-1i*omegam1*a2-1/2*omegam1.^2*b2^2)-1);% regime 2
% Psi_p2=-1i*omegap1*mu2-1/2*omegap1.^2*sigma2^2+lambda2*(p2*exp(-1i*omegap1*a1-1/2*omegap1.^2*b1^2)+(1-p2)*exp(-1i*omegap1*a2-1/2*omegap1.^2*b2^2)-1);
% Epsilon_diag_m=zeros(2,2,n_r+1);Epsilon_diag_p=zeros(2,2,n_r+1);
% Epsilon_diag_m(1,1,:)=Psi_m1;Epsilon_diag_m(2,2,:)=Psi_m2;
% Epsilon_diag_p(1,1,:)=Psi_p1;Epsilon_diag_p(2,2,:)=Psi_p2;
% 
% Epsilon_diag_m=Epsilon_diag_m+Q';Epsilon_diag_p=Epsilon_diag_p+Q';
% Eig_m_1=1/2*(Epsilon_diag_m(1,1,:)+Epsilon_diag_m(2,2,:)-sqrt(Epsilon_diag_m(1,1,:).^2+Epsilon_diag_m(2,2,:).^2-...
%     2*Epsilon_diag_m(1,1,:).*Epsilon_diag_m(2,2,:)+4*Epsilon_diag_m(1,2,:).*Epsilon_diag_m(2,1,:)));
% Eig_m_2=1/2*(Epsilon_diag_m(1,1,:)+Epsilon_diag_m(2,2,:)+sqrt(Epsilon_diag_m(1,1,:).^2+Epsilon_diag_m(2,2,:).^2-...
%     2*Epsilon_diag_m(1,1,:).*Epsilon_diag_m(2,2,:)+4*Epsilon_diag_m(1,2,:).*Epsilon_diag_m(2,1,:)));
% Eig_p_1=1/2*(Epsilon_diag_p(1,1,:)+Epsilon_diag_p(2,2,:)-sqrt(Epsilon_diag_p(1,1,:).^2+Epsilon_diag_p(2,2,:).^2-...
%     2*Epsilon_diag_p(1,1,:).*Epsilon_diag_p(2,2,:)+4*Epsilon_diag_p(1,2,:).*Epsilon_diag_p(2,1,:)));
% Eig_p_2=1/2*(Epsilon_diag_p(1,1,:)+Epsilon_diag_p(2,2,:)+sqrt(Epsilon_diag_p(1,1,:).^2+Epsilon_diag_p(2,2,:).^2-...
%     2*Epsilon_diag_p(1,1,:).*Epsilon_diag_p(2,2,:)+4*Epsilon_diag_p(1,2,:).*Epsilon_diag_p(2,1,:)));
% Epsilon_m=exp(Eig_m_1*delta_t).*(eye(2,2)+zeros(2,2,n_grid+1))+(exp(Eig_m_1*delta_t)-exp(Eig_m_2*delta_t))./(Eig_m_1-Eig_m_2).*(Epsilon_diag_m-Eig_m_1.*(eye(2,2)+zeros(2,2,n_grid+1)));
% Epsilon_p=exp(Eig_p_1*delta_t).*(eye(2,2)+zeros(2,2,n_grid+1))+(exp(Eig_p_1*delta_t)-exp(Eig_p_2*delta_t))./(Eig_p_1-Eig_p_2).*(Epsilon_diag_p-Eig_p_1.*(eye(2,2)+zeros(2,2,n_grid+1)));
% 
% n_v=2;
% Chara_m=zeros(n_l+n_r+1,n_v,n_v,'gpuArray');
% Chara_p=zeros(n_l+n_r+1,n_v,n_v,'gpuArray');
% Chara_m(n_l+1:end,:,:)=permute(Epsilon_m,[3,2,1]);
% Chara_m(1:n_l,:,:)=conj(Chara_m(2*n_l+1:-1:n_l+2,:,:));
% Chara_p(n_l+1:end,:,:)=permute(Epsilon_p,[3,2,1]);
% Chara_p(1:n_l,:,:)=conj(Chara_p(2*n_l+1:-1:n_l+2,:,:));
% w=ones(n_v,1);


% toc;
% tic;
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


x_0=-log((N_monitor+1)*K./S_0-1); % fusai 2016

ww=[1/2,ones(1,n_r)];

%% Fourier inversion
W_hat_m_01=1/(2*pi)*(Chara_m(n_l+1:end,:,1).*T_m(n_l+1:end,:))*delta_omega*w;
V_p_m_1=exp(-r*Tmat)/(N_monitor+1)/pi*real(max((N_monitor+1)*K-S_0,0).*exp(-1i*x_0*omegam1.').*ww*(W_hat_m_01.*Gammainvm)*delta_omega);
V_c_m_1=(V_p_m_1+exp(-r*Tmat)*(S_0/(N_monitor+1)*(1-exp((r-q)*(N_monitor+1)*delta_t))/(1-exp((r-q)*delta_t))-K)); % option price with initial state 1
% Re_error_call_1=abs(V_c_m_1-V_c_benchmark_1)./V_c_benchmark_1;

% W_hat_m_02=1/(2*pi)*(Chara_m(n_l+1:end,:,2).*T_m(n_l+1:end,:))*delta_omega*w;
% V_p_m_2=exp(-r*Tmat)/(N_monitor+1)/pi*real(max((N_monitor+1)*K-S_0,0).*exp(-1i*x_0*omegam1.').*ww*(W_hat_m_02.*Gammainvm)*delta_omega);
% V_c_m_2=(V_p_m_2+exp(-r*Tmat)*(S_0/(N_monitor+1)*(1-exp((r-q)*(N_monitor+1)*delta_t))/(1-exp((r-q)*delta_t))-K)); % option price with initial state 2
% Re_error_call_2=abs(V_c_m_2-V_c_benchmark_2)./V_c_benchmark_2;


W_hat_p_01=(Chara_p(n_l+1:end,:,1).*(W_p(n_l+1:end,:)+1/(2*pi)*T_p(n_l+1:end,:)*delta_omega))*w; 
V_p_p_1=exp(-r*Tmat)/(N_monitor+1)/pi*real(max((N_monitor+1)*K-S_0,0).*exp(-1i*x_0*omegap1.').*ww*(W_hat_p_01.*Gammainvp)*delta_omega);
V_c_p_1=(V_p_p_1+exp(-r*Tmat)*(S_0/(N_monitor+1)*(1-exp((r-q)*(N_monitor+1)*delta_t))/(1-exp((r-q)*delta_t))-K)); % option price with initial state 1
% 
% W_hat_p_02=(Chara_p(n_l+1:end,:,2).*(W_p(n_l+1:end,:)+1/(2*pi)*T_p(n_l+1:end,:)*delta_omega))*w;
% V_p_p_2=exp(-r*Tmat)/(N_monitor+1)/pi*real(max((N_monitor+1)*K-S_0,0).*exp(-1i*x_0*omegap1.').*ww*(W_hat_p_02.*Gammainvp)*delta_omega);
% V_c_p_2=(V_p_p_2+exp(-r*Tmat)*(S_0/(N_monitor+1)*(1-exp((r-q)*(N_monitor+1)*delta_t))/(1-exp((r-q)*delta_t))-K)); % option price with initial state 2

%% Greeks
% Delta_p_m=-V_p_m_1./((N_monitor+1).*K-S_0)-exp(-r*Tmat)*K/(pi*S_0).*real(1i*omegam1.'.*exp(-1i*x_0*omegam1.').*ww*(W_hat_m_01.*Gammainvm)*delta_omega);
% Delta_c_m=Delta_p_m+exp(-r*Tmat)/(N_monitor+1)*(1-exp((r-q)*(N_monitor+1)*delta_t))/(1-exp((r-q)*delta_t));
% Gamma_p_m=exp(-r*Tmat)*(N_monitor+1)*K.^2./(pi*S_0^2.*((N_monitor+1)*K-S_0)).*real(1i*omegam1.'.*(1+1i*omegam1.').*exp(-1i*x_0*omegam1.').*ww*(W_hat_m_01.*Gammainvm)*delta_omega);
% Gamma_c_m=Gamma_p_m;

%% Richardson extrapolation
% V_c_500=(342*72*V_c_12-(342*475+18*247)*V_c_25+(342*650+76*247)*V_c_50)/(247*400); % V_c_N represent option prices with the number of monitoring points N, N=12, 25, 50




toc;
