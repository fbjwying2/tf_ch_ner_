import java.io.OutputStream;
import java.io.InputStream;
import java.net.Socket;

public class NerJavaClicent {	
    public static void main(String []args) throws Exception {		
		String HOST = "127.0.0.1";
		int PORT = 21567;
		int BUFSIZ = 1024;
		Socket socket = new Socket(HOST, PORT);
		OutputStream outputStream = socket.getOutputStream();
		
		long startTime = System.currentTimeMillis();
		outputStream.write(("请问佛山在哪里办理身份证").getBytes());
		outputStream.flush();
		System.out.println(socket);
		
		InputStream is = socket.getInputStream();
		byte[] bytes = new byte[BUFSIZ];
		int n = is.read(bytes);
		String result_str = new String(bytes, 0, n, "utf-8");
		String clientStr = new String(result_str.getBytes("GBK"),"GBK");
		
		long endTime = System.currentTimeMillis();
		
		System.out.println(clientStr);
		System.out.println("running time:" + (endTime - startTime) + "ms");
		
		is.close();
		socket.close();
    }
}