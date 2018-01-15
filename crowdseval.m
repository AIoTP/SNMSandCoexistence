load('coexistence.mat')
load('crowds_face_val.mat')
load('hr.mat')
crowds_face_boxes=norm_score(crowds_face_boxes);
crowds_face_boxes1=norm_score(crowds_face_boxes1);%%coexistence
facehr=zeros(1000,2);
facehr1=zeros(1000,2);
IoU_thresh = 0.5;
%统计TP,NP,goundtruth
ggpp=0;
for i=1:12
    face=crowds_face_boxes{i};
    face0=crowds_face_val{i};
    face0(:,3)=face0(:,1)+face0(:,3);
    face0(:,4)=face0(:,2)+face0(:,4);
    face0=single(face0);
    face(:,1:4)=ceil(face(:,1:4));
  for t = 1:1000
    jilu=facehr(t,:); 
    thresh = 1-t/1000; 
    recall_list=zeros(size(face0,1),1);
     for h = 1:size(face,1)
          if face(h,5)>=thresh
             overlap_list = boxoverlap(face0, face(h,1:4));
             [max_overlap, idx] = max(overlap_list);
             if (max_overlap >= IoU_thresh)&&(recall_list(idx)==0)
               recall_list(idx) = 1;
               jilu(1)=jilu(1)+1;
             else
               jilu(2)=jilu(2)+1;  
             end
          end
     end
      facehr(t,:)=jilu; 
  end
  ggpp=size(face0,1)+ggpp;
end

Y=zeros(1000,1);
facehrr=zeros(1000,2);
for t = 1:1000
     Y(t)=1-t/1000;
     if facehr(t,1)+facehr(t,2)~=0
         facehrr(t,1)=facehr(t,1)/(facehr(t,1)+facehr(t,2));
         facehrr(t,2)=facehr(t,1)/ggpp;
     end
end
%统计TP,NP,goundtruth coexistence
ggpp=0;
for i=1:12
    face1=crowds_face_boxes1{i};
    face0=crowds_face_val{i};
    face0(:,3)=face0(:,1)+face0(:,3);
    face0(:,4)=face0(:,2)+face0(:,4);
    face0=single(face0);
    face1(:,1:4)=ceil(face1(:,1:4));
  for t = 1:1000
    jilu1=facehr1(t,:); 
    thresh = 1-t/1000; 
    recall_list=zeros(size(face0,1),1);
     for h = 1:size(face1,1)
          if face1(h,5)>=thresh
             overlap_list = boxoverlap(face0, face1(h,1:4));
             [max_overlap, idx] = max(overlap_list);
             if (max_overlap >= IoU_thresh)&&(recall_list(idx)==0)
               recall_list(idx) = 1;
               jilu1(1)=jilu1(1)+1;
             else
               jilu1(2)=jilu1(2)+1;  
             end
          end
     end
      facehr1(t,:)=jilu1; 
  end
  ggpp=size(face0,1)+ggpp;
end

Y=zeros(1000,1);
facehrr1=zeros(1000,2);
for t = 1:1000
     Y(t)=1-t/1000;
     if facehr1(t,1)+facehr1(t,2)~=0
         facehrr1(t,1)=facehr1(t,1)/(facehr1(t,1)+facehr1(t,2));
         facehrr1(t,2)=facehr1(t,1)/ggpp;
     end
end
%%%%%画图
plot(facehrr(:,2),facehrr(:,1),'LineWidth',1.5,'Color',[0 1 1]);
grid on;
hold on;
plot(facehrr1(:,2),facehrr1(:,1),'LineWidth',1.5,'Color',[1 0 1]);
xlim([0,1]);
ylim([0,1]);
legend('Noncoexistence','Coexistence');
xlabel('Recall');
ylabel('Precision');
hold on;
