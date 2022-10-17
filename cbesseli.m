
function Inu = cbesseli(nu,x)
 %-----   Calculate gamma function with a complex order   ----
nu=reshape(nu,[],1);
x=reshape(x,[],1);
Inu=zeros(length(nu),length(x));
real_index=find(imag(nu)==0);
complex_index=find(imag(nu)~=0);
if ~isempty(real_index)
    Inu(real_index,:)=besseli(nu(real_index).*ones(1,length(x)),ones(length(real_index),1).*x.',1);
end
Inu(complex_index,:)=(1i).^(-nu(complex_index)).*cbesselj(nu(complex_index),1i*x); 

end