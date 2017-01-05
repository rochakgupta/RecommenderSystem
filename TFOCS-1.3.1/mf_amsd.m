function mf_amsd
    ind=[20,40,60,80,100];
    a=zeros(5,12);
    for i=1:6
        arr = mf_amsd_calc(i);
        a(:,2*i-1) = arr(:,1); 
        a(:,2*i) = arr(:,2);
    end;
    display(a);
    dlmwrite('finalamsd.csv',a,',');
end
        
        