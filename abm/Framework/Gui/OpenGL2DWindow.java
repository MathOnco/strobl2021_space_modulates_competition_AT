package Framework.Gui;
import Framework.Interfaces.ColorIntGenerator;
import Framework.Util;
import org.lwjgl.BufferUtils;
import org.lwjgl.LWJGLException;
import org.lwjgl.opengl.Display;
import org.lwjgl.opengl.DisplayMode;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;

import static org.lwjgl.opengl.GL11.*;
import static Framework.Util.*;

/**
 * Created by rafael on 5/28/17.
 */
public class OpenGL2DWindow {
    final boolean active;
    final public int xPix;
    final public int yPix;
    final public int xDim;
    final public int yDim;
    TickTimer tt = new TickTimer();
    final float[] circlPtsDefault = Util.GenCirclePoints(1, 20);

    /**
     * creates a new OpenGL2DWindow
     * @param title the title that will appear at the top of the window (default "")
     * @param xPix the length of the window in screen pixels
     * @param yPix the height of the window in screen pixels
     * @param xDim the length that the window will represent for drawing, should match the xDim of the model
     * @param yDim the height that the window will represent for drawing, should match the yDim of the model
     * @param active if set to false, the OpenGL2DWindow will not actually render and its methods will be skipped (default true)
     */
    public OpenGL2DWindow(String title, int xPix, int yPix, int xDim, int yDim, boolean active) {
        this.active = active;
        this.xPix = xPix;
        this.yPix = yPix;
        this.xDim = xDim;
        this.yDim = yDim;
        if (active) {
            try {
                Display.setDisplayMode(new DisplayMode(xPix, yPix));
                Display.setTitle(title);
                Display.create();
            } catch (LWJGLException e) {
                e.printStackTrace();
                System.err.println("unable to create Vis3D display");
            }
            glMatrixMode(GL_PROJECTION);
            glLoadIdentity();
            glOrtho(0, xDim, 0, yDim, -1, 1);
            glMatrixMode(GL_MODELVIEW);
        }
    }

    /**
     * the below constructors are variants of the above constructor with default values for some of the arguments
     */
    public OpenGL2DWindow(int xPix, int yPix, int xDim, int yDim) {
        this("", xPix, yPix, xDim, yDim, true);
    }
    public OpenGL2DWindow(int xPix, int yPix, int xDim, int yDim, boolean active) {
        this("", xPix, yPix, xDim, yDim, active);
    }
    public OpenGL2DWindow(String title,int xPix, int yPix, int xDim, int yDim) {
        this(title, xPix, yPix, xDim, yDim, true);
    }

    /**
     * call this once per step of your model, and the function will ensure that your model runs at the rate provided in
     * millis. the function will take the amount time between calls into account to ensure a consistent tick rate.
     */
    public void TickPause(int millis) {
        if (active) {
            tt.TickPause(millis);
        }
    }

    /**
     * usually called before any other draw commands, sets the screen to a color.
     */
    public void Clear(int clearColor) {
        if (active) {
            glClearColor((float) GetRed(clearColor), (float) GetGreen(clearColor), (float) GetBlue(clearColor), (float) GetAlpha(clearColor));
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        }
    }

    /**
     * renders all draw commands to the window
     */
    public void Update() {
        if (active) {
            Display.update();
        }
    }

    /**
     * returns true if the close button has been clicked in the Gui
     */
    public boolean IsClosed() {
        if (active) {
            return Display.isCloseRequested();
        } else {
            return true;
        }
    }

    /**
     * closes the gui
     */
    public void Close() {
        if (active) {
            Display.destroy();
        }
    }

    /**
     * returns whether the Gui is active (whether it exists)
     */
    public boolean IsActive() {
        return active;
    }

    /**
     * draws a circle centered around x,y, subsequent circles will be drawn over top
     */
    public void Circle(double x, double y, double rad, int color) {
        FanShape((float) x, (float) y, (float) rad, circlPtsDefault, color);
    }

    /**
     * draws a circle centered around x,y, subsequent circles will be drawn over top, the ColorGen function is used to
     * generate the color of the circle and will not be called if the Gui is not active
     */
    public void Circle(double x, double y, double rad, ColorIntGenerator ColorGen) {
        if (active) {
            FanShape((float) x, (float) y, (float) rad, circlPtsDefault, ColorGen.GenColorInt());
        }
    }

    /**
     * draws a fan shape around the center x and y, using the array of points to define the edges of the fan. used as
     * part of the circle function
     */
    public void FanShape(float centerX, float centerY, float scale, float[] points, ColorIntGenerator ColorGen) {
        if (active) {
            FanShape(centerX, centerY, scale, points, ColorGen.GenColorInt());
        }
    }

    /**
     * draws a fan shape around the center x and y, using the array of points to define the edges of the fan. used as
     * part of the circle function
     */
    public void FanShape(float centerX, float centerY, float scale, float[] points, int color) {
        if (active) {
            glColor4f((float) GetRed(color), (float) GetGreen(color), (float) GetBlue(color), (float) GetAlpha(color));
            glBegin(GL_TRIANGLE_FAN);
            glVertex2f(centerX, centerY);
            for (int i = 0; i < points.length / 2; i++) {
                glVertex2f((points[i * 2] * scale + centerX), (points[i * 2 + 1] * scale + centerY));
            }
            glEnd();
        }
    }

    /**
     * draws a line between (x1,y1) and (x2,y2)
     */
    public void Line(double x1, double y1, double x2, double y2, int color) {
        if (active) {
            glColor4f((float) GetRed(color), (float) GetGreen(color), (float) GetBlue(color), (float) GetAlpha(color));
            glBegin(GL_LINES);
            glVertex2f((float) x1, (float) y1);
            glVertex2f((float) x2, (float) y2);
            glEnd();
        }
    }

    /**
     * draws a series of lines between all x,y pairs
     */
    public void LineStrip(double[] xs, double[] ys, int color) {
        if (active) {
            glColor4f((float) GetRed(color), (float) GetGreen(color), (float) GetBlue(color), (float) GetAlpha(color));
            glBegin(GL_LINE_STRIP);
            for (int i = 0; i < xs.length; i++) {
                glVertex2f((float) xs[i], (float) ys[i]);
            }
            glEnd();
        }
    }

    /**
     * draws a series of lines between all x,y pairs, coords should store pairs as x,y,x,y...
     */
    public void LineStrip(double[] coords, int color) {
        if (active) {
            glColor4f((float) GetRed(color), (float) GetGreen(color), (float) GetBlue(color), (float) GetAlpha(color));
            glBegin(GL_LINE_STRIP);
            for (int i = 0; i < coords.length; i += 2) {
                glVertex2f((float) coords[i], (float) coords[i + 1]);
            }
            glEnd();
        }
    }

    /**
     * saves the current state image to a PNG, call Update first
     */
    public void ToPNG(String path) {
        SaveImg(path, "png");
    }

    /**
     * saves the current state image to a JPG, call Update first
     */
    public void ToJPG(String path) {
        SaveImg(path, "jpg");
    }

    /**
     * saves the current state image to a GIF image, call Update first
     */
    public void ToGIF(String path) {
        SaveImg(path, "gif");
    }

    void SaveImg(String path, String mode) {
        if (active) {
            File out = new File(path);
            glReadBuffer(GL_FRONT);
            int width = Display.getDisplayMode().getWidth();
            int height = Display.getDisplayMode().getHeight();
            int bpp = 4; // Assuming a 32-bit display with a byte each for red, green, blue, and alpha.
            ByteBuffer buffer = BufferUtils.createByteBuffer(width * height * bpp);
            glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, buffer);
            BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);

            for (int x = 0; x < width; x++) {
                for (int y = 0; y < height; y++) {
                    int i = (x + (width * y)) * bpp;
                    int r = buffer.get(i) & 0xFF;
                    int g = buffer.get(i + 1) & 0xFF;
                    int b = buffer.get(i + 2) & 0xFF;
                    image.setRGB(x, height - (y + 1), (0xFF << 24) | (r << 16) | (g << 8) | b);
                }
            }
            try {
                ImageIO.write(image, mode, out);
            } catch (IOException e) {
                e.printStackTrace();
            }

        }
    }
}
