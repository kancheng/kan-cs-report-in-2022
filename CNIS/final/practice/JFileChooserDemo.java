import java.awt.*;
import java.awt.event.*;
import javax.swing.*;
import java.io.*;
@SuppressWarnings("serial")
public class JFileChooserDemo extends JFrame implements ActionListener
{
   JFileChooser fc = new JFileChooser();//创建文件对话框对象
   JButton open,save;
   public JFileChooserDemo() {
      Container container = this.getContentPane();
      container.setLayout(new FlowLayout());
      this.setTitle("文件对话框演示程序");
      open = new JButton("打开文件");
      save = new JButton("保存文件");
      open.addActionListener(this);
      save.addActionListener(this);
      container.add(open);//添加到内容窗格上
      container.add(save);
      this.setVisible(true);
      this.setSize(600,450);
   }
   
   public void actionPerformed(ActionEvent e) {
       JButton button = (JButton)e.getSource(); // 得到事件源
       if(button == open) // 选择的是“打开文件”按钮
       {
           int select = fc.showOpenDialog(this); // 显示打开文件对话框
           if(select == JFileChooser.APPROVE_OPTION) // 选择的是否为“确认”
           {
               File file = fc.getSelectedFile();
               System.out.println("文件"+file.getName()+"被打开");
           }
           else
               System.out.println("打开操作被取消"); // 在屏幕上输出
       }
       
      if(button == save) { // 选择的是“保存文件”按钮
      
           int select = fc.showSaveDialog(this); // 显示保存文件对话框
           if (select == JFileChooser.APPROVE_OPTION) {
               File file = fc.getSelectedFile();
               System.out.println( "文件" + file.getName() + "被保存");
           } else {
               System.out.println("保存操作被取消");
           }

      }
   }
   public static void main(String[] args) {
       new JFileChooserDemo();
   }
}
