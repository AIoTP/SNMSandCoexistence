function boxes = coexistence2(boxes,prob_thresh)
prob_thresh1=-log(1/prob_thresh-1);
%load nummmm;
tishen=zeros(600,1);
for i=1:size(boxes,1)   
            areahhh=(boxes(i,3)-boxes(i,1))*(boxes(i,4)-boxes(i,2));  
            areahhh=ceil(areahhh);
         for j=1:600
             if areahhh==j&&(boxes(i,end)>0)
                 tishen(j)=tishen(j)+1;
             end
         end
end
ll=size(boxes,1);
i=1;
while i<=ll    
       areahhh=(boxes(i,3)-boxes(i,1))*(boxes(i,4)-boxes(i,2));
       areahhh=ceil(areahhh);
     if 0<areahhh&&areahhh<600 
       j1=max(0.9*areahhh,1);
       j1=ceil(j1);
       j2=min(1.1*areahhh,600);
       j2=ceil(j2);
       fff=0;
       for j=j1:j2
         fff=fff+tishen(j);
       end
         if fff>10
         ww=0.5+1/(1+exp(-fff));
         boxes(i,end)=ww*(boxes(i,end)+3.47)-3.47;
         end   
     end
     if boxes(i,end)<prob_thresh1
        boxes(i,:)=[];
        ll=ll-1;
     else
         i=i+1;
     end
end









