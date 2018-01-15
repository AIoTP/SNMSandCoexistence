function boxes = facecoexistence(boxes,overlap,prob_thresh)

scores_thresh=-log(1/prob_thresh-1);
jiaochaa=0.5;
jiaocha=-log(1/jiaochaa-1);
score_overlap=-log(1/overlap-1);

%if ~exist('use_gpu', 'var')
%    use_gpu = false;
%end

%if use_gpu
%   s = boxes(:, end);
%    if ~issorted(s(end:-1:1))
%       [~, I] = sort(s, 'descend');
%        boxes = boxes(I, :);
%        pick = nms_gpu_mex(single(boxes)', double(overlap));
%       pick = I(pick);
%    else
%       pick = nms_gpu_mex(single(boxes)', double(overlap));
%    end
 %   return;
%end
    
%if size(boxes, 1) < 1000000
%    pick = soft-nms_mex(double(boxes), double(overlap));
%    return;
%end
%%%%%%%%%%%%%%%%%%%%%%%%%�ȶ�boxes���֣����Ŷȣ��Ӵ�С����
s = boxes(:,end);

[~, I] = sort(s,'descend');
boxes=boxes(I,:);
%�ӷִ�С������boxes��i,:��
ll=length(I);
i=1;
while i<ll 
 %if boxes(i,end)<0
  %   break;
 %end
 x1 = boxes(i,1);
 y1 = boxes(i,2);
 x2 = boxes(i,3);
 y2 = boxes(i,4);
 
 areai = (x2-x1+1) .* (y2-y1+1);
 j=i+1;
%����boxes��i,:���������п��ҵ��غ��������overlap������
 while j<=ll
   jx1 = boxes(j,1);
   jy1 = boxes(j,2);
   jx2 = boxes(j,3);
   jy2 = boxes(j,4);
   areaj = (jx2-jx1+1) .* (jy2-jy1+1);
  xx1 = max(x1, jx1);
  yy1 = max(y1, jy1);
  xx2 = min(x2, jx2);
  yy2 = min(y2, jy2);
  
  w = max(0.0, xx2-xx1+1);
  h = max(0.0, yy2-yy1+1);
  %����غϵ��������
  inter = w.*h;
  o = inter ./ (areai + areaj - inter);
%������ص���boxes(I(j),[1:4 end])���ر�ͣ���ô���������غ������intert������areai,areaj�ı�ֵ
%������overlap,��������ɾ��������overlapȡ0��nms
%�����˴�С�򹲴棬���Ǵ�С�������棬��һ���nmsȥ�����龰�����⣬�Ҽ���ɾȥ�Ŀ�����С���Աȿ����ִ��������������ɾ��
   if (boxes(j,end)<0)&&(o>0)
      %areabei=areai/areaj;
      interi=inter/areai;
      interj=inter/areaj;
      %if (interi>overlap)||( interj>overlap)
      if (interj>0.9)&&(boxes(i,end)>5)
      boxes(j,:)=[];
      ll=ll-1;
      end
  end
    j=j+1;   
end
 
%boxes(i,:)������е�boxes(j,:)���ж��غ��������Ӧ�Ĵ������
%������һ��boxes(i,:)��
i=i+1;

end
end














