import java.awt.Dimension;
import java.awt.Toolkit;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.ComponentAdapter;
import java.awt.event.ComponentEvent;

import javax.swing.JButton;
import javax.swing.JFrame;

public class JFTestJava extends JFrame implements ActionListener{

    public  JButton btnNew;
    public  JButton btnEnd;
    public JFTestJava() {
        setLayout(null);
        setVisible(true);
        Toolkit tk = Toolkit.getDefaultToolkit();
        Dimension d = tk.getScreenSize();
        int w, h;
        w = d.width;
        h = d.height;
        setSize( w, h);
        btnNew = new JButton("New");
        btnEnd = new JButton("Exit");
        btnNew.addActionListener(this);
        btnEnd.addActionListener(this);
        add(btnNew);
        add(btnEnd);
        action_viewchange(d);
    }  

    public void action_viewchange(Dimension dimen){
        /*  setSize(int width, int height) */
        int w, h;
        w =dimen.width;
        h =dimen.height;
        setSize( w, h);
        btnNew.setBounds( w/2 - w/10, h/3, w/5, h/10);
        btnEnd.setBounds( w/2 + w/10, h/3, w/5, h/10);
        
    }

    public void actionPerformed(ActionEvent e){
        if( e.getSource() == btnNew){
            /* 新增視窗 */
            new JFTestJava();
        } else if(e.getSource() == btnEnd){
            exit();
        }
    }    
    public void exit(){
        System.exit(0);
    }
   // btnNew.setBounds( w/2 - w/10, h/3, w/5, h/10);
   // btnEnd.setBounds( w/2 + w/10, h/3, w/5, h/10);

    public static void main(String arg[]){
        JFTestJava t = new JFTestJava();
        t.addComponentListener(new ComponentAdapter() {
            public void componentResized(ComponentEvent e) {
                    Dimension sizeval = e.getComponent().getSize();
                    System.out.println("Size Changed : " + sizeval);
                    t.action_viewchange(sizeval);
                    t.revalidate();
            }
        });
    }
}