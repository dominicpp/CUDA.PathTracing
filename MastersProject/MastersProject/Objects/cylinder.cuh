//#pragma once
//
//#include "../Hit/hit.cuh"
//#include "../Ray/ray.cuh"
//
//class Material;
//
//class Cylinder : public Hit {
//	Vec3 m_position;
//	float m_radius, m_height;
//	Material* m_material;
//
//public:
//	Cylinder() = default;
//	Cylinder(Vec3 position, float radius, float height, Material* material) : m_position(position), m_radius(radius), m_height(height), m_material(material) {};
//
//	__host__ __device__ virtual bool hitIntersect(const Ray& ray, float tmin, float tmax, RecordHit& hit) const override;
//};
//
//bool Cylinder::hitIntersect(const Ray& ray, float tmin, float tmax, RecordHit& hit) const
//{
//	Vec3 newPosition = ray.getOrigin() - m_position;
//	float a = pow(ray.getDirection().a[0], 2) + pow(ray.getDirection().a[2], 2);
//	float b = 2 * (newPosition.a[0] * ray.getDirection().a[0] + newPosition.a[2] * ray.getDirection().a[2]);
//	float c = pow(newPosition.a[0], 2) + pow(newPosition.a[2], 2) - pow(m_radius, 2);
//	float discriminant = pow(b, 2) - (4 * a * c);
//	
//	if (discriminant < 0) return false;
//	if (discriminant >= 0)
//	{
//		float t1 = (-(b + sqrt(discriminant))) / (2 * a);
//		float t2 = (-(b - sqrt(discriminant))) / (2 * a);
//		float t = t1;
//
//		if (t1 > t2) t1 = t2;
//		t2 = t;
//
//		double y1 = newPosition.a[1] + t1 * ray.getDirection().a[1];
//		double y2 = newPosition.a[1] + t2 * ray.getDirection().a[1];
//
//		if (y1 < -m_height) {
//			if (y2 < -m_height)
//				return NULL;
//			else {
//				double t = t1 + (t2 - t1) * (y1 + m_height) / (y1 - y2);
//				if (!(tmin < t && t < tmax)) {
//					return NULL;
//				}
//				else {
//					Vec3 hitVec = ray.pointAt(t);
//					Vec3 hitNormVec = (hitVec - m_position) / m_radius;
//					hit.rayParameter = t;
//					hit.material = m_material;
//					return true;
//				}
//
//			}
//		}
//
//		if (y1 >= -m_height && y1 <= m_height) {
//			if (!(tmin < t && t < tmax)) {
//				return NULL;
//			}
//			else {
//				Vec3 hitVec = ray.pointAt(t1);
//				Vec3 hitNormVec = (hitVec - m_position) / m_radius;
//				hit.rayParameter = t;
//				hit.material = m_material;
//				return true;
//			}
//		}
//
//		if (y1 > m_height) {
//			if (y2 > m_height) {
//				return NULL;
//			}
//			else {
//				double t = t1 + (t2 - t1) * (y1 - m_height) / (y1 - y2);
//				if (!(tmin < t && t < tmax)) {
//					return NULL;
//				}
//				else {
//					Vec3 hitVec = ray.pointAt(t);
//					Vec3 hitNormVec = (hitVec - m_position) / m_radius;
//					hit.rayParameter = t;
//					hit.material = m_material;
//					return true;
//				}
//			}
//		}
//	}
//}