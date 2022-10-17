% function Jnu = cbesselj(nu,x)
% %  -----   Calculate gamma function with a complex order   ----
% 
% kMax = 100;
% tol = 1e-30;
% Nnu=length(nu);
% Nx = length(x);
% nu=reshape(nu,[],1);
% x=reshape(x,[],1);
% 
% Jnu=zeros(Nnu,Nx);
% re_index=find(imag(nu)==0);
% im_index=find(imag(nu)~=0);
% Jnu(re_index,:)=besselj(nu(re_index)*ones(1,Nx),ones(length(re_index),1)*x.');
% nu_immatrix=nu(im_index)*ones(1,Nx);
% x_matrix=ones(length(im_index),1)*x.';
% Jnu_immatrix=Jnu(im_index,:);
% Im_index=((1:length(im_index)*Nx)');
% k_index=Im_index;
% Index_allnaninf=[];
% for k=0:kMax    
%     Jnu_Im=Jnu_immatrix(k_index);
%     JnuLast=Jnu_immatrix(k_index);
%     nu_k=nu_immatrix(k_index);
%     x_k=x_matrix(k_index);
%     Jnu_Im = Jnu_Im + (-0.25*x_k.*x_k).^k./(gamma(k+1)*cgamma(nu_k+k+1));
%         
%     Index_naninf=isnan(Jnu_Im)+isinf(abs(Jnu_Im));
%     Index_isnaninf=find(Index_naninf==1);
%     Index_nonaninf=find(Index_naninf==0);
%     Index_allnaninf=sort([Index_allnaninf;k_index(Index_isnaninf)]);
%     Jnu_Im=Jnu_Im(Index_nonaninf);
%     JnuLast=JnuLast(Index_nonaninf);
%     Jnu_immatrix(k_index(Index_nonaninf))=Jnu_Im;
%     RI=find(real(Jnu_Im)~=0 & imag(Jnu_Im)~=0);
%     R=find(real(Jnu_Im)~=0 & imag(Jnu_Im)==0);
%     k_index(Index_isnaninf)=[];
%     Rerr=abs((real(Jnu_Im(RI))-real(JnuLast(RI)))./real(Jnu_Im(RI)));
%     Ierr=abs((imag(Jnu_Im(RI))-imag(JnuLast(RI)))./imag(Jnu_Im(RI)));
%     RRerr=abs((real(Jnu_Im(R))-real(JnuLast(R)))./real(Jnu_Im(R)));
%     index1=find(Rerr<=tol & Ierr<=tol);
%     index2=find(RRerr<=tol);
%     RI(index1)=[];
%     R(index2)=[];
%     kR=k_index(sort([RI;R]));
%     if ~isempty(kR)
%         k_index=Im_index(kR);
%     else
%         break;
%     end
% end
% 
% Jnu_immatrix = Jnu_immatrix.*(0.5*x_matrix).^nu_immatrix;
% Jnu(im_index,:)=Jnu_immatrix;
% if ~isempty(Index_allnaninf)
%     math('matlab2math','nu',nu_immatrix(Index_allnaninf));
%     math('matlab2math','xm',x_matrix(Index_allnaninf));
%     Jnu(Index_allnaninf)=math('math2matlab','BesselJ[nu,xm]');
% end
% if k==kMax
%     disp('Algorithm does not converge in the calculation of bessel function. Maximum concurence number arrived!');
% end
% 
% end   

function Jnu = cbesselj(nu,x)
%  -----   Calculate gamma function with a complex order   ----

kMax = 100;

nu=reshape(nu,[],1);
x=reshape(x,[],1);
Nnu=length(nu);
Nx = length(x);
% tol = 1e-30;
% Jnu=zeros(Nnu,Nx,'gpuArray');
Jnu=zeros(Nnu,Nx);
re_index=find(imag(nu)==0);
im_index=find(imag(nu)~=0);
if ~isempty(re_index)
    Jnu(re_index,:)=besselj(nu(re_index)*ones(1,Nx),ones(length(re_index),1)*x.',1);
end
nu_immatrix=(nu(im_index)*ones(1,Nx));
x_matrix=(ones(length(im_index),1)*x.');
tol = exp(-abs(imag(x_matrix)))*1e-10;
Jnu_immatrix=Jnu(im_index,:);
% Im_index=gpuArray((1:length(im_index)*Nx)');
Im_index=((1:length(im_index)*Nx)');
k_index=Im_index;
Index_allnaninf=[];
for k=0:kMax    
    Jnu_Im=Jnu_immatrix(k_index);
    JnuLast=Jnu_immatrix(k_index);
    nu_k=nu_immatrix(k_index);
    x_k=x_matrix(k_index);
    Jnu_Im = Jnu_Im + exp(-abs(imag(x_k))).*(-0.25*x_k.*x_k).^k./(gamma(k+1)*cgamma(nu_k+k+1));
        
    Index_naninf=isnan(Jnu_Im)+isinf(abs(Jnu_Im));
    Index_isnaninf=find(Index_naninf==1);
    Index_nonaninf=find(Index_naninf==0);
    Index_allnaninf=sort([Index_allnaninf;k_index(Index_isnaninf)]);
    Jnu_Im=Jnu_Im(Index_nonaninf);
    JnuLast=JnuLast(Index_nonaninf);
    Jnu_immatrix(k_index(Index_nonaninf))=Jnu_Im;
    RI=find(real(Jnu_Im)~=0 & imag(Jnu_Im)~=0);
    R=find(real(Jnu_Im)~=0 & imag(Jnu_Im)==0);
    k_index(Index_isnaninf)=[];
    Rerr=abs((real(Jnu_Im(RI))-real(JnuLast(RI)))./real(Jnu_Im(RI)));
    Ierr=abs((imag(Jnu_Im(RI))-imag(JnuLast(RI)))./imag(Jnu_Im(RI)));
    RRerr=abs((real(Jnu_Im(R))-real(JnuLast(R)))./real(Jnu_Im(R)));
    tol_RI=tol(k_index(RI));
    tol_R=tol(k_index(R));
    index1=find(Rerr<=tol_RI & Ierr<=tol_RI);
    index2=find(RRerr<=tol_R);
    RI(index1)=[];
    R(index2)=[];
    kR=k_index(sort([RI;R]));
    if ~isempty(kR)
        k_index=Im_index(kR);
    else
        break;
    end
end

Jnu_immatrix = Jnu_immatrix.*(0.5*x_matrix).^nu_immatrix;
Jnu(im_index,:)=Jnu_immatrix;
if ~isempty(Index_allnaninf)
    math('matlab2math','nu',nu_immatrix(Index_allnaninf));
    math('matlab2math','xm',x_matrix(Index_allnaninf));
    math('math2matlab','Expxm=Exp[-Abs[Im[xm]]]');
    Jnu(Index_allnaninf)=math('math2matlab','Expxm*BesselJ[nu,xm]+0.I');
end
if k==kMax
    disp('Algorithm does not converge in the calculation of bessel function. Maximum concurence number arrived!');
end

end   