import javax.swing.*;
import javax.swing.event.*;
import java.awt.*;
import java.awt.event.*;
import java.util.Random;
public class Mondecaro extends JFrame implements ActionListener {
    private JButton button;
    private JPanel panel;
    private int height = 300, width = 300;
    public static void main (String[] argv) {
        Mondecaro frame = new Mondecaro();
        frame.setSize(520, 350);
        frame.createGUI();
        frame.setVisible(true);
    }
    private void createGUI() {
        setDefaultCloseOperation(EXIT_ON_CLOSE);
        Container window = getContentPane();
        window.setLayout(new FlowLayout());
        panel = new JPanel();
        panel.setPreferredSize(new Dimension(width, height));
        panel.setBackground(Color.white);
        window.add(panel);
        button = new JButton("Mondecaro");
        window.add(button);
        button.addActionListener(this);
    }
    public void actionPerformed(ActionEvent e) {
        draw();
    }
    private void draw() {
        Graphics paper = panel.getGraphics();
        paper.setColor(Color.white);
        paper.fillRect( 0, 0, width, height);
        paper.setColor(Color.black);
        int px = width, py = height / 2;
        Random random = new Random ();
        int n = width / 2, in = 0, N = 1000;
        for (int i = 1; i <= N; i++) {
            int x = random.nextInt(width);
            int y = random.nextInt(width);
            double dist = Math.sqrt(Math.pow(x - n, 2) + Math.pow(y - n, 2));
            if (dist < n) {
                paper.setColor(Color.red);
                in++;
            } else {
                paper.setColor(Color.blue);
            }
            paper.drawOval( x, y, 1, 1);
            setTitle("Area = " + (double)in / i * 4);
            for (int k = 0; k < 1000; k++); 
        }
    }
}